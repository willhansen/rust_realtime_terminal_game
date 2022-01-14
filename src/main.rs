extern crate line_drawing;
extern crate nalgebra;
extern crate num;
extern crate std;
extern crate termion;

#[macro_use]
extern crate more_asserts;

use nalgebra::{point, vector, Point2, Vector2};
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;
use termion::color;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

// const player_jump_height: i32 = 3;
// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: i32 = 30; // frames per second

// a block every two ticks
const PLAYER_DEFAULT_MAX_SPEED_BPS: f32 = 30.0; // blocks per second
const PLAYER_DEFAULT_MAX_SPEED_BPF: f32 = PLAYER_DEFAULT_MAX_SPEED_BPS / MAX_FPS as f32; // blocks per frame
const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Block {
    None,
    Wall,
    Brick,
    Player,
}

impl Block {
    fn glyph(&self) -> char {
        match self {
            Block::None => ' ',
            Block::Wall => '█',
            Block::Brick => '▪',
            Block::Player => '█',
        }
    }

    fn color_str(&self) -> String {
        match self {
            Block::Player => format!(
                "{}{}",
                color::Fg(color::Red).to_string(),
                color::Bg(color::Black).to_string()
            ),
            _ => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
                color::Bg(color::Black).to_string()
            ),
        }
    }

    fn subject_to_normal_gravity(&self) -> bool {
        match self {
            Block::None | Block::Wall | Block::Player => false,
            _ => true,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MovecastCollision {
    pos: Point2<i32>,
    normal: Vector2<i32>,
}

struct Game {
    grid: Vec<Vec<Block>>,      // (x,y), left to right, top to bottom
    prev_grid: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    terminal_size: (u16, u16),  // (width, height)
    prev_mouse_pos: (i32, i32), // where mouse was last frame (if pressed)
    // last_pressed_key: Option<termion::event::Key>,
    running: bool,         // set false to quit
    selected_block: Block, // What the mouse places
    player_alive: bool,
    player_pos: (i32, i32),
    player_x_max_speed_bpf: f32,
    player_x_vel_bpf: f32,
    player_y_vel_bpf: f32,
    player_desired_x_direction: i32,
    player_accumulated_x_err: f32, // speed can be a float
    player_accumulated_y_err: f32, // speed can be a float
    player_acceleration_from_gravity: f32,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            grid: vec![vec![Block::None; height as usize]; width as usize],
            prev_grid: vec![vec![Block::None; height as usize]; width as usize],
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            // last_pressed_key: None,
            running: true,
            selected_block: Block::Wall,
            player_alive: false,
            player_pos: (0, 0),
            player_x_max_speed_bpf: PLAYER_DEFAULT_MAX_SPEED_BPF,
            player_x_vel_bpf: 0.0,
            player_y_vel_bpf: 0.0,
            player_desired_x_direction: 0,
            player_accumulated_x_err: 0.0,
            player_accumulated_y_err: 0.0,
            player_acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
        }
    }

    fn screen_to_world(&self, terminal_position: &(u16, u16)) -> (i32, i32) {
        // terminal indexes from 1, and the y axis goes top to bottom
        (
            terminal_position.0 as i32 - 1,
            self.terminal_size.1 as i32 - terminal_position.1 as i32,
        )
    }

    fn world_to_screen(&self, world_position: &(i32, i32)) -> (u16, u16) {
        // terminal indexes from 1, and the y axis goes top to bottom
        // world indexes from 0, origin at bottom left
        (
            world_position.0 as u16 + 1,
            (self.terminal_size.1 as i32 - world_position.1) as u16,
        )
    }
    fn get_block(&self, pos: (i32, i32)) -> Block {
        return self.grid[pos.0 as usize][pos.1 as usize];
    }

    fn draw_line(&mut self, pos0: (i32, i32), pos1: (i32, i32), block: Block) {
        for (x1, y1) in line_drawing::Bresenham::new(pos0, pos1) {
            self.grid[x1 as usize][y1 as usize] = block;
        }
    }

    fn draw_point(&mut self, pos: (i32, i32), block: Block) {
        self.grid[pos.0 as usize][pos.1 as usize] = block;
    }

    // todo update
    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![Block::None; height as usize]; width as usize];
    }

    fn place_player(&mut self, x: i32, y: i32) {
        // Need to kill existing player if still alive
        if self.player_alive {
            self.kill_player();
        }
        self.grid[x as usize][y as usize] = Block::Player;
        self.player_x_vel_bpf = 0.0;
        self.player_y_vel_bpf = 0.0;
        self.player_desired_x_direction = 0;
        self.player_pos = (x, y);
        self.player_alive = true;
    }

    // When The player presses the jump button
    fn player_jump(&mut self) {
        self.player_y_vel_bpf = 1.0;
        // TODO
    }

    fn player_set_desired_x_direction(&mut self, new_x_dir: i32) {
        if new_x_dir != self.player_desired_x_direction {
            self.player_desired_x_direction = new_x_dir.signum();
            self.player_accumulated_x_err = 0.0;
        }
    }

    fn handle_input(&mut self, evt: termion::event::Event) {
        match evt {
            Event::Key(ke) => match ke {
                Key::Char('q') => self.running = false,
                Key::Char('1') => self.selected_block = Block::None,
                Key::Char('2') => self.selected_block = Block::Wall,
                Key::Char('3') => self.selected_block = Block::Brick,
                Key::Char('c') => self.clear(),
                Key::Char(' ') => self.player_jump(),
                Key::Char('a') | Key::Left => self.player_set_desired_x_direction(-1),
                Key::Char('s') | Key::Down => self.player_set_desired_x_direction(0),
                Key::Char('d') | Key::Right => self.player_set_desired_x_direction(1),
                _ => {}
            },
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.draw_point((x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Press(MouseButton::Right, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_player(x, y);
                    self.draw_point((x, y), Block::Player);
                }
                MouseEvent::Hold(term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.draw_line(self.prev_mouse_pos, (x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn tick_physics(&mut self) {
        self.apply_gravity();
        if self.player_alive {
            self.apply_player_motion();
        }
    }

    fn update_screen(
        &mut self,
        stdout: &mut MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    ) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if self.grid[x][y] != self.prev_grid[x][y] {
                    let (term_x, term_y) = self.world_to_screen(&(x as i32, y as i32));
                    if self.grid[x][y] == Block::Player {
                        write!(
                            stdout,
                            "{}{}{}{}",
                            termion::cursor::Goto(term_x, term_y),
                            color::Fg(color::Red).to_string(),
                            self.grid[x][y].glyph(),
                            color::Fg(color::White).to_string(),
                        )
                        .unwrap();
                    } else {
                        write!(
                            stdout,
                            "{}{}",
                            termion::cursor::Goto(term_x, term_y),
                            self.grid[x][y].glyph(),
                        )
                        .unwrap();
                    }
                }
            }
        }
        write!(stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        stdout.flush().unwrap();
        self.prev_grid = self.grid.clone();
    }

    fn apply_gravity(&mut self) {
        for x in 0..self.terminal_size.0 as usize {
            for y in 0..self.terminal_size.1 as usize {
                // We want to count from bottom to top, because things fall down
                let block = self.grid[x as usize][y];
                if block.subject_to_normal_gravity() {
                    let is_bottom_row = y == 0;
                    let has_direct_support = !is_bottom_row && self.grid[x][y - 1] != Block::None;
                    if is_bottom_row {
                        self.grid[x][y] = Block::None;
                    } else if !has_direct_support {
                        self.grid[x][y - 1] = block;
                        self.grid[x][y] = Block::None;
                    }
                }
            }
        }
    }

    fn move_player_to(&mut self, x: i32, y: i32) {
        self.grid[self.player_pos.0 as usize][self.player_pos.1 as usize] = Block::None;
        self.grid[x as usize][y as usize] = Block::Player;
        self.player_pos = (x, y);
    }

    fn kill_player(&mut self) {
        self.grid[self.player_pos.0 as usize][self.player_pos.1 as usize] = Block::None;
        self.player_alive = false;
    }

    fn apply_player_acceleration_from_traction(&mut self) {
        let acceleration_from_traction = 1.0;
        let start_x_vel = self.player_x_vel_bpf;
        let desired_acceleration_direction = self.player_desired_x_direction.signum();

        let trying_to_stop = desired_acceleration_direction == 0 && start_x_vel != 0.0;
        let started_above_max_speed = start_x_vel.abs() > self.player_x_max_speed_bpf;

        let real_acceleration_direction;
        if trying_to_stop || started_above_max_speed {
            real_acceleration_direction = -start_x_vel.signum() as i32;
        } else {
            real_acceleration_direction = desired_acceleration_direction;
        }
        let delta_vx = real_acceleration_direction.signum() as f32 * acceleration_from_traction;
        let mut end_x_vel = self.player_x_vel_bpf + delta_vx;
        let changed_direction = start_x_vel * end_x_vel < 0.0;

        if trying_to_stop && changed_direction {
            end_x_vel = 0.0;
        }

        let want_to_go_faster =
            start_x_vel.signum() == desired_acceleration_direction.signum() as f32;
        let ended_below_max_speed = end_x_vel.abs() < self.player_x_max_speed_bpf;
        if started_above_max_speed && ended_below_max_speed && want_to_go_faster {
            end_x_vel = end_x_vel.signum() * self.player_x_max_speed_bpf;
        }
        // if want go fast, but starting less than max speed.  Can't go past max speed.
        if !started_above_max_speed && !ended_below_max_speed {
            end_x_vel = end_x_vel.signum() * self.player_x_max_speed_bpf;
        }

        if end_x_vel == 0.0 {
            self.player_accumulated_x_err = 0.0; // TODO: double check this
        }
        self.player_x_vel_bpf = end_x_vel;
    }

    fn apply_player_acceleration_from_gravity(&mut self) {
        self.player_y_vel_bpf -= self.player_acceleration_from_gravity;
    }

    // including_gravity
    fn apply_player_motion(&mut self) {
        if !self.player_is_supported() {
            self.apply_player_acceleration_from_gravity();
        }
        self.apply_player_acceleration_from_traction();

        // let x_dir: i32 = signum(self.player_desired_x_direction);
        // instant acceleration
        // self.player_x_vel_bpf = self.player_x_max_speed_bpf * x_dir as f32;

        let dx_ideal: f32 = self.player_x_vel_bpf + self.player_accumulated_x_err;
        let dx_actual: i32 = dx_ideal.trunc() as i32;
        self.player_accumulated_x_err = dx_ideal.fract();

        let dy_ideal: f32 = self.player_y_vel_bpf + self.player_accumulated_y_err;
        let dy_actual: i32 = dy_ideal.trunc() as i32;
        self.player_accumulated_y_err = dy_ideal.fract();

        let target_x = self.player_pos.0 + dx_actual;
        let target_y = self.player_pos.1 + dy_actual;

        // need to check intermediate blocks for being clear
        if let Some(collision) = self.movecast(self.player_pos, (target_x, target_y)) {
            let (x, y) = (collision.pos.x, collision.pos.y);
            self.move_player_to(x, y);
            if collision.normal.x != 0 {
                // hit an obstacle and lose velocity
                self.player_accumulated_x_err = 0.0;
                self.player_x_vel_bpf = 0.0;
            }
            if collision.normal.y != 0 {
                self.player_accumulated_y_err = 0.0;
                self.player_y_vel_bpf = 0.0;
            }
        } else {
            if !self.in_world(target_x, target_y) {
                // Player went out of bounds and died
                self.kill_player();
            } else {
                // no collision, and in world
                self.move_player_to(target_x, target_y);
            }
        }
    }

    // Where the player can move to in a line
    // tries to draw a line in air
    // returns None if out of bounds
    // returns the start position if start is not Block::None
    fn movecast(&self, start: (i32, i32), end: (i32, i32)) -> Option<MovecastCollision> {
        let mut prev_pos = start;
        for pos in line_drawing::WalkGrid::new(start, end) {
            if self.in_world(pos.0, pos.1) {
                if self.get_block(pos) != Block::None && self.get_block(pos) != Block::Player {
                    // one before the block
                    return Some(MovecastCollision {
                        pos: Point2::new(prev_pos.0, prev_pos.1),
                        normal: Vector2::new(prev_pos.0 - pos.0, prev_pos.1 - pos.1),
                    });
                } else if pos == end {
                    // if we made it to the end, no collision
                    return None;
                }
            } else {
                // ran off the grid.  No collision
                return None;
            }
            // the prev_pos will be the start pos for the first 2 run throughs of the loop
            prev_pos = pos;
        }
        // shouldn't get here
        return None;
    }

    fn player_is_supported(&self) -> bool {
        self.player_pos.1 > 0
            && self.grid[self.player_pos.0 as usize][self.player_pos.1 as usize - 1] != Block::None
    }

    fn in_world(&self, x: i32, y: i32) -> bool {
        return x >= 0
            && x < self.terminal_size.0 as i32
            && y >= 0
            && y < self.terminal_size.1 as i32;
    }
}

fn main() {
    let stdin = stdin();
    let (width, height) = termion::terminal_size().unwrap();
    let mut game = Game::new(width, height);
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    write!(
        stdout,
        "{}{}q to exit.  c to clear.  Mouse to draw.  Begin!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    stdout.flush().unwrap();

    let (tx, rx) = channel();

    // Separate thread for reading input
    thread::spawn(move || {
        for c in stdin.events() {
            let evt = c.unwrap();
            tx.send(evt).unwrap();
        }
    });

    // time saver
    game.draw_line(
        ((width / 5) as i32, (height / 4) as i32),
        ((4 * width / 5) as i32, (height / 4) as i32),
        Block::Wall,
    );

    while game.running {
        while let Ok(evt) = rx.try_recv() {
            game.handle_input(evt);
        }
        game.tick_physics();
        game.update_screen(&mut stdout);
        // TODO: make better
        thread::sleep(Duration::from_millis(16));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_placement_and_gravity() {
        let mut game = Game::new(30, 30);
        game.draw_point((10, 10), Block::Wall);
        game.draw_point((20, 10), Block::Brick);

        assert_eq!(game.grid[10][10], Block::Wall);
        assert_eq!(game.grid[20][10], Block::Brick);
        game.tick_physics();

        assert_eq!(game.grid[10][10], Block::Wall);
        assert_eq!(game.grid[20][10], Block::None);

        assert_eq!(game.grid[10][9], Block::None);
        assert_eq!(game.grid[20][9], Block::Brick);
    }

    #[test]
    fn test_place_player() {
        let mut game = Game::new(30, 30);
        // TODO: should these be variables?  Or should I just hardcode them?
        let (x1, y1, x2, y2) = (15, 11, 12, 5);
        assert_eq!(game.player_alive, false);
        game.place_player(x1, y1);

        assert_eq!(game.grid[x1 as usize][y1 as usize], Block::Player);
        assert_eq!(game.player_pos, (x1, y1));
        assert_eq!(game.player_alive, true);

        game.place_player(x2, y2);
        assert_eq!(game.grid[x1 as usize][y1 as usize], Block::None);
        assert_eq!(game.grid[x2 as usize][y2 as usize], Block::Player);
        assert_eq!(game.player_pos, (x2, y2));
        assert_eq!(game.player_alive, true);
    }

    #[test]
    fn test_player_dies_when_falling_off_screen() {
        let mut game = Game::new(30, 30);
        game.place_player(15, 1);
        game.player_acceleration_from_gravity = 10.0;
        game.tick_physics();
        assert_eq!(game.player_alive, false);
    }

    #[test]
    fn test_movecast() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);

        assert_eq!(
            game.movecast((15, 9), (15, 11)),
            Some(MovecastCollision {
                pos: point![15, 9],
                normal: vector![0, -1]
            })
        );
        assert_eq!(
            game.movecast((15, 10), (15, 11)),
            Some(MovecastCollision {
                pos: point![15, 10],
                normal: vector![0, 0]
            })
        );
        assert_eq!(
            game.movecast((15, 9), (17, 11)),
            Some(MovecastCollision {
                pos: point![15, 9],
                normal: vector![0, -1]
            })
        );
        assert_eq!(
            game.movecast((15, 9), (17, 110)),
            Some(MovecastCollision {
                pos: point![15, 9],
                normal: vector![0, -1]
            })
        );
        assert_eq!(game.movecast((150, 9), (17, 11)), None);
        assert_eq!(game.movecast((1, 9), (-17, 9)), None);
        assert_eq!(game.movecast((15, 9), (17, -11)), None);
    }

    #[test]
    fn test_movecast_ignore_player() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);

        assert_eq!(game.movecast((15, 11), (15, 13)), None);
    }
    #[test]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.in_world(0, 0));
        assert!(game.in_world(29, 29));
        assert!(!game.in_world(30, 30));
        assert!(!game.in_world(10, -1));
        assert!(!game.in_world(-1, 10));
    }

    #[test]
    fn move_player() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 1.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert_eq!(game.player_pos, (16, 11));
        assert_eq!(game.grid[15][11], Block::None);
        assert_eq!(game.grid[16][11], Block::Player);

        game.place_player(15, 11);
        assert_eq!(game.player_desired_x_direction, 0);
        game.player_desired_x_direction = -1;

        game.tick_physics();
        game.tick_physics();

        assert_eq!(game.grid[15][11], Block::None);
        assert_eq!(game.grid[13][11], Block::Player);
        assert_eq!(game.player_pos, (13, 11));
    }
    #[test]
    fn test_stop_on_collision() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.draw_point((16, 11), Block::Wall);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 1.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert_eq!(game.player_pos, (15, 11));
        assert_eq!(game.grid[16][11], Block::Wall);
        assert_eq!(game.grid[15][11], Block::Player);
        assert_eq!(game.player_x_vel_bpf, 0.0);
    }

    #[test]
    fn move_player_slowly() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 0.5;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert_eq!(game.player_pos, (15, 11));

        game.tick_physics();

        assert_eq!(game.player_pos, (16, 11));
    }
    #[test]
    fn test_move_player_quickly() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 2.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();
        game.tick_physics();
        assert_gt!(game.player_pos.0, 17);
        game.tick_physics();
        assert_gt!(game.player_pos.0, 19);
    }
    #[test]
    fn test_fast_player_collision_between_frames() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        // Player should not teleport through this block
        game.draw_point((16, 11), Block::Wall);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 2.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();
        assert_eq!(game.player_pos, (15, 11));
        assert_eq!(game.player_x_vel_bpf, 0.0);
    }
    #[test]
    fn test_can_jump() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);

        game.place_player(15, 11);

        game.player_jump();
        game.tick_physics();
        assert_eq!(game.player_pos, (15, 12));
    }
    #[test]
    fn test_player_gravity() {
        let mut game = Game::new(30, 30);
        game.player_acceleration_from_gravity = 1.0;
        game.place_player(15, 11);

        game.tick_physics();
        game.tick_physics();

        assert_lt!(game.player_pos.1, 11);
    }

    #[test]
    fn test_slide_on_collision() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.draw_line((14, 10), (14, 20), Block::Wall);

        game.place_player(15, 11);
        game.player_x_vel_bpf = -2.0;
        game.player_y_vel_bpf = 2.0;
        game.tick_physics();
        assert_eq!(game.player_x_vel_bpf, 0.0);
        assert_gt!(game.player_y_vel_bpf, 0.0);
    }
    #[test]
    fn test_decellerate_when_already_moving_faster_than_max_speed() {
        let mut game = Game::new(30, 30);
        game.place_player(15, 11);
        game.player_x_max_speed_bpf = 1.0;
        game.player_x_vel_bpf = 5.0;
        game.player_desired_x_direction = 1;
        game.tick_physics();
        assert_gt!(game.player_x_vel_bpf, game.player_x_max_speed_bpf);
        assert_lt!(game.player_x_vel_bpf, 5.0);
    }
}
