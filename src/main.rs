extern crate line_drawing;
extern crate nalgebra;
extern crate num;
extern crate std;
extern crate termion;

// use assert2::{assert, check};
use nalgebra::{point, vector, Point2, Vector2};
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use termion::color;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

// const player_jump_height: i32 = 3;
// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: i32 = 60; // frames per second
const IDEAL_FRAME_DURATION_MS: u128 = (1000.0 / MAX_FPS as f32) as u128;

// a block every two ticks
const PLAYER_DEFAULT_MAX_SPEED_BPS: f32 = 30.0; // blocks per second
const PLAYER_DEFAULT_MAX_SPEED_BPF: f32 = PLAYER_DEFAULT_MAX_SPEED_BPS / MAX_FPS as f32; // blocks per frame
const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;
const DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION: f32 = 1.0;
const DEFAULT_PLAYER_COYOTE_TIME_DURATION_S: f32 = 0.2;
const DEFAULT_PLAYER_MAX_COYOTE_FRAMES: i32 =
    ((DEFAULT_PLAYER_COYOTE_TIME_DURATION_S * MAX_FPS as f32) + 1.0) as i32;

// "heighth", "reighth"
const EIGHTH_BLOCKS_FROM_LEFT: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
const EIGHTH_BLOCKS_FROM_BOTTOM: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

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
            Block::Player => EIGHTH_BLOCKS_FROM_LEFT[8],
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        match self {
            Block::None | Block::Wall | Block::Player => false,
            _ => true,
        }
    }
    fn wall_grabbable(&self) -> bool {
        match self {
            Block::None => false,
            _ => true,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MovecastCollision {
    pos: Point2<i32>,
    normal: Vector2<i32>,
}

struct Player {}

struct ColoredGlyph {
    glyph: char,
    fg_color: Option<String>,
    bg_color: Option<String>,
}

impl ColoredGlyph {
    fn to_string(&self) -> String {
        return format!(
            "{}{}{}{}{}",
            self.fg_color.clone().unwrap(),
            self.bg_color.clone().unwrap(),
            self.glyph,
            color::Fg(color::Reset).to_string(),
            color::Bg(color::Reset).to_string()
        );
    }
}

struct Game {
    grid: Vec<Vec<Block>>,              // (x,y), left to right, top to bottom
    grid_at_last_draw: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    output_buffer: Vec<Vec<char>>,      // (x,y), left to right, top to bottom
    output_on_screen: Vec<Vec<char>>,   // (x,y), left to right, top to bottom
    terminal_size: (u16, u16),          // (width, height)
    prev_mouse_pos: (i32, i32),         // where mouse was last frame (if pressed)
    // last_pressed_key: Option<termion::event::Key>,
    running: bool,         // set false to quit
    selected_block: Block, // What the mouse places
    player_alive: bool,
    player_pos: Point2<i32>,
    player_max_run_speed_bpf: f32,
    player_vel_bpf: Vector2<f32>,
    player_desired_x_direction: i32,
    player_accumulated_pos_err: Vector2<f32>, // speed can be a float
    player_acceleration_from_gravity: f32,
    player_acceleration_from_traction: f32,
    player_remaining_coyote_frames: i32,
    player_max_coyote_frames: i32,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            grid: vec![vec![Block::None; height as usize]; width as usize],
            grid_at_last_draw: vec![vec![Block::None; height as usize]; width as usize],
            output_buffer: vec![vec![' '; height as usize]; width as usize],
            output_on_screen: vec![vec![' '; height as usize]; width as usize],
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
            player_alive: false,
            player_pos: point![0, 0],
            player_max_run_speed_bpf: PLAYER_DEFAULT_MAX_SPEED_BPF,
            player_vel_bpf: Vector2::<f32>::new(0.0, 0.0),
            player_desired_x_direction: 0,
            player_accumulated_pos_err: Vector2::<f32>::new(0.0, 0.0),
            player_acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
            player_acceleration_from_traction: DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION,
            player_remaining_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
            player_max_coyote_frames: DEFAULT_PLAYER_MAX_COYOTE_FRAMES,
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
    fn get_block(&self, pos: Point2<i32>) -> Block {
        return self.grid[pos.x as usize][pos.y as usize];
    }
    fn set_block(&mut self, pos: Point2<i32>, block: Block) {
        self.grid[pos.x as usize][pos.y as usize] = block;
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
        self.player_vel_bpf = vector![0.0, 0.0];
        self.player_desired_x_direction = 0;
        self.player_pos = point![x, y];
        self.player_alive = true;
        self.player_remaining_coyote_frames = 0;
    }
    // When The player presses the jump button
    fn player_jump(&mut self) {
        let jump_delta_v = 1.0;
        if self.player_is_grabbing_wall() {
            self.player_vel_bpf.x += jump_delta_v * -self.player_wall_grab_direction() as f32;
            self.player_desired_x_direction *= -1;
        }
        self.player_vel_bpf.y = jump_delta_v;
    }
    fn player_jump_if_possible(&mut self) {
        if self.player_is_supported() || self.player_is_grabbing_wall() {
            self.player_jump();
        }
    }

    fn player_set_desired_x_direction(&mut self, new_x_dir: i32) {
        if new_x_dir != self.player_desired_x_direction {
            self.player_desired_x_direction = new_x_dir.signum();
            self.player_accumulated_pos_err.x = 0.0;
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
                Key::Char('r') => self.place_player(
                    self.terminal_size.0 as i32 / 2,
                    self.terminal_size.1 as i32 / 2,
                ),
                Key::Char(' ') => self.player_jump_if_possible(),
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
        self.apply_gravity_to_blocks();
        if self.player_alive {
            self.apply_player_motion();
        }
    }

    fn update_output_buffer(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for y in 0..height {
                self.output_buffer[x][y] = self.grid[x][y].glyph();
            }
        }

        let player_glyphs = self.get_player_glyphs();
        let x_offset;
        if self.player_accumulated_pos_err.x < 0.0 {
            x_offset = -1;
        } else {
            x_offset = 0;
        }

        for i in 0..player_glyphs.len() {
            let x = self.player_pos.x + i as i32 + x_offset;
            let y = self.player_pos.y;
            if self.in_world(point![x, y]) {
                self.output_buffer[x as usize][y as usize] = player_glyphs[i];
            }
        }
    }
    fn get_player_glyphs(&self) -> Vec<char> {
        let subsquare_position_in_eighths =
            (self.player_accumulated_pos_err.x * 8.0).trunc() as i32;
        assert!(subsquare_position_in_eighths.abs() <= 8);
        if subsquare_position_in_eighths < 0 {
            return vec![
                EIGHTH_BLOCKS_FROM_LEFT[(8 - subsquare_position_in_eighths.abs()) as usize],
                EIGHTH_BLOCKS_FROM_LEFT[subsquare_position_in_eighths.abs() as usize],
            ];
        } else {
            return vec![
                EIGHTH_BLOCKS_FROM_LEFT[(8 - subsquare_position_in_eighths) as usize],
                EIGHTH_BLOCKS_FROM_LEFT[subsquare_position_in_eighths as usize],
            ];
        }
    }

    fn get_buffered_glyph(&self, pos: Point2<i32>) -> char {
        return self.output_buffer[pos.x as usize][pos.y as usize];
    }
    fn get_glyph_on_screen(&self, pos: Point2<i32>) -> char {
        return self.output_on_screen[pos.x as usize][pos.y as usize];
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
                if self.output_buffer[x][y] != self.output_on_screen[x][y] {
                    let (term_x, term_y) = self.world_to_screen(&(x as i32, y as i32));
                    let is_player_square =
                        self.player_alive && point![x as i32, y as i32] == self.player_pos;
                    write!(stdout, "{}", termion::cursor::Goto(term_x, term_y)).unwrap();
                    if is_player_square {
                        write!(stdout, "{}", color::Fg(color::Red).to_string()).unwrap();
                    }
                    write!(stdout, "{}", self.output_buffer[x][y]).unwrap();
                    if is_player_square {
                        write!(stdout, "{}", color::Fg(color::Reset).to_string()).unwrap();
                    }
                }
            }
        }
        write!(stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        stdout.flush().unwrap();
        self.output_on_screen = self.output_buffer.clone();
    }

    fn apply_gravity_to_blocks(&mut self) {
        // We want to count from bottom to top, because things fall down
        for x in 0..self.terminal_size.0 as usize {
            for y in 0..self.terminal_size.1 as usize {
                let block = self.grid[x][y];
                if block.subject_to_block_gravity() {
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

    fn move_player_to(&mut self, pos: Point2<i32>) {
        self.set_block(self.player_pos, Block::None);
        self.set_block(pos, Block::Player);
        self.player_pos = pos;
    }

    fn kill_player(&mut self) {
        self.set_block(self.player_pos, Block::None);
        self.player_alive = false;
    }

    fn player_is_grabbing_wall(&self) -> bool {
        if self.player_desired_x_direction != 0 && self.player_vel_bpf.y <= 0.0 {
            if let Some(block) = self
                .get_block_relative_to_player(vector![self.player_desired_x_direction.signum(), 0])
            {
                return block.wall_grabbable();
            }
        }
        return false;
    }
    fn player_wall_grab_direction(&self) -> i32 {
        // TODO: is this good?
        if self.player_is_grabbing_wall() {
            return self.player_desired_x_direction.signum();
        } else {
            return 0;
        }
    }

    fn apply_player_acceleration_from_wall_friction(&mut self) {
        let direction_of_acceleration = -self.player_vel_bpf.y.signum();
        let delta_v = direction_of_acceleration * self.player_acceleration_from_traction;
        if delta_v.abs() > self.player_vel_bpf.y.abs() {
            self.player_vel_bpf.y = 0.0;
        } else {
            self.player_vel_bpf.y += delta_v;
        }
    }

    fn apply_player_acceleration_from_floor_traction(&mut self) {
        let start_x_vel = self.player_vel_bpf.x;
        let desired_acceleration_direction = self.player_desired_x_direction.signum();

        let trying_to_stop = desired_acceleration_direction == 0 && start_x_vel != 0.0;
        let started_above_max_speed = start_x_vel.abs() > self.player_max_run_speed_bpf;

        let real_acceleration_direction;
        if trying_to_stop || started_above_max_speed {
            real_acceleration_direction = -start_x_vel.signum() as i32;
        } else {
            real_acceleration_direction = desired_acceleration_direction;
        }
        let delta_vx =
            real_acceleration_direction.signum() as f32 * self.player_acceleration_from_traction;
        let mut end_x_vel = self.player_vel_bpf.x + delta_vx;
        let changed_direction = start_x_vel * end_x_vel < 0.0;

        if trying_to_stop && changed_direction {
            end_x_vel = 0.0;
        }

        let want_to_go_faster =
            start_x_vel.signum() == desired_acceleration_direction.signum() as f32;
        let ended_below_max_speed = end_x_vel.abs() < self.player_max_run_speed_bpf;
        if started_above_max_speed && ended_below_max_speed && want_to_go_faster {
            end_x_vel = end_x_vel.signum() * self.player_max_run_speed_bpf;
        }
        // if want go fast, but starting less than max speed.  Can't go past max speed.
        if !started_above_max_speed && !ended_below_max_speed {
            end_x_vel = end_x_vel.signum() * self.player_max_run_speed_bpf;
        }

        if end_x_vel == 0.0 {
            self.player_accumulated_pos_err.x = 0.0; // TODO: double check this
        }
        self.player_vel_bpf.x = end_x_vel;
    }

    fn apply_player_acceleration_from_gravity(&mut self) {
        self.player_vel_bpf.y -= self.player_acceleration_from_gravity;
    }
    // including_gravity
    fn apply_player_motion(&mut self) {
        self.update_player_acceleration();

        // let x_dir: i32 = signum(self.player_desired_x_direction);
        // instant acceleration
        // self.player_x_vel_bpf = self.player_max_run_speed_bpf * x_dir as f32;

        let ideal_step: Vector2<f32> = self.player_vel_bpf + self.player_accumulated_pos_err;
        let attempted_step: Vector2<i32> = trunc(ideal_step);
        self.player_accumulated_pos_err = fract(ideal_step);

        let target = self.player_pos + attempted_step;

        let step_taken: Vector2<i32>;
        // need to check intermediate blocks for being clear
        if let Some(collision) = self.movecast(self.player_pos, target) {
            step_taken = collision.pos - self.player_pos;
            self.move_player_to(collision.pos);
            if collision.normal.x != 0 {
                // hit an obstacle and lose velocity
                self.player_accumulated_pos_err.x = 0.0;
                self.player_vel_bpf.x = 0.0;
            }
            if collision.normal.y != 0 {
                self.player_accumulated_pos_err.y = 0.0;
                self.player_vel_bpf.y = 0.0;
            }
        } else {
            if !self.in_world(target) {
                // Player went out of bounds and died
                self.kill_player();
                return;
            } else {
                // no collision, and in world
                step_taken = target - self.player_pos;
                self.move_player_to(target);
            }
        }

        // moved vertically => instant empty charge
        if step_taken.y != 0 {
            self.player_remaining_coyote_frames = 0;
        }
        if self.player_is_standing_on_block() {
            self.player_remaining_coyote_frames = self.player_max_coyote_frames;
        } else if self.player_remaining_coyote_frames > 0 {
            self.player_remaining_coyote_frames -= 1;
        }
    }
    fn update_player_acceleration(&mut self) {
        if self.player_is_grabbing_wall() && self.player_vel_bpf.y <= 0.0 {
            self.apply_player_acceleration_from_wall_friction();
        } else {
            if !self.player_is_supported() {
                self.apply_player_acceleration_from_gravity();
            }
            self.apply_player_acceleration_from_floor_traction();
        }
    }

    // Where the player can move to in a line
    // tries to draw a line in air
    // returns None if out of bounds
    // returns the start position if start is not Block::None
    fn movecast(&self, start: Point2<i32>, end: Point2<i32>) -> Option<MovecastCollision> {
        let mut prev_pos = start;
        for (x, y) in line_drawing::WalkGrid::new((start.x, start.y), (end.x, end.y)) {
            let pos = point![x, y];
            if self.in_world(pos) {
                if self.get_block(pos) != Block::None && self.get_block(pos) != Block::Player {
                    // one before the block
                    return Some(MovecastCollision {
                        pos: prev_pos,
                        normal: prev_pos - pos,
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
        return self.player_is_standing_on_block() || self.player_remaining_coyote_frames > 0;
    }

    fn player_is_standing_on_block(&self) -> bool {
        match self.get_block_below_player() {
            None | Some(Block::None) => false,
            _ => true,
        }
    }

    fn get_block_below_player(&self) -> Option<Block> {
        return self.get_block_relative_to_player(vector![0, -1]);
    }

    fn get_block_relative_to_player(&self, rel_pos: Vector2<i32>) -> Option<Block> {
        let target_pos = self.player_pos + rel_pos;
        if self.player_alive && self.in_world(target_pos) {
            return Some(self.get_block(target_pos));
        }
        return None;
    }

    fn in_world(&self, pos: Point2<i32>) -> bool {
        return pos.x >= 0
            && pos.x < self.terminal_size.0 as i32
            && pos.y >= 0
            && pos.y < self.terminal_size.1 as i32;
    }
    fn init_world(&mut self) {
        let bottom_left = (
            (self.terminal_size.0 / 5) as i32,
            (self.terminal_size.1 / 4) as i32,
        );
        self.draw_line(
            bottom_left,
            ((4 * self.terminal_size.0 / 5) as i32, bottom_left.1),
            Block::Wall,
        );
        self.draw_line(
            bottom_left,
            (bottom_left.0, 3 * (self.terminal_size.1 / 4) as i32),
            Block::Wall,
        );
        self.place_player(
            self.terminal_size.0 as i32 / 2,
            self.terminal_size.1 as i32 / 2,
        );

        self.player_acceleration_from_traction = 0.3;
        self.player_max_run_speed_bpf = 0.5
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
    game.init_world();

    while game.running {
        let start_time = Instant::now();
        while let Ok(evt) = rx.try_recv() {
            game.handle_input(evt);
        }
        game.tick_physics();
        game.update_output_buffer();
        game.update_screen(&mut stdout);
        let tick_duration_so_far_ms = start_time.elapsed().as_millis();
        if tick_duration_so_far_ms < IDEAL_FRAME_DURATION_MS {
            thread::sleep(Duration::from_millis(
                (IDEAL_FRAME_DURATION_MS - tick_duration_so_far_ms) as u64,
            ));
        }
    }
}

fn trunc(vec: Vector2<f32>) -> Vector2<i32> {
    return Vector2::<i32>::new(vec.x.trunc() as i32, vec.y.trunc() as i32);
}

fn fract(vec: Vector2<f32>) -> Vector2<f32> {
    return Vector2::<f32>::new(vec.x.fract(), vec.y.fract());
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    fn p(x: i32, y: i32) -> Point2<i32> {
        return point![x, y];
    }
    fn v<T>(x: T, y: T) -> Vector2<T> {
        return vector![x, y];
    }

    fn set_up_player_on_platform() -> Game {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        return game;
    }

    fn set_up_player_hanging_on_wall_on_left() -> Game {
        let mut game = Game::new(30, 30);
        game.draw_line((14, 0), (14, 20), Block::Wall);
        game.place_player(15, 11);
        game.player_desired_x_direction = -1;
        return game;
    }

    #[test]
    fn test_placement_and_gravity() {
        let mut game = Game::new(30, 30);
        game.draw_point((10, 10), Block::Wall);
        game.draw_point((20, 10), Block::Brick);

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::Brick);
        game.tick_physics();

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::None);

        assert!(game.grid[10][9] == Block::None);
        assert!(game.grid[20][9] == Block::Brick);
    }

    #[test]
    fn test_place_player() {
        let mut game = Game::new(30, 30);
        // TODO: should these be variables?  Or should I just hardcode them?
        let (x1, y1, x2, y2) = (15, 11, 12, 5);
        assert!(game.player_alive == false);
        game.place_player(x1, y1);

        assert!(game.grid[x1 as usize][y1 as usize] == Block::Player);
        assert!(game.player_pos == p(x1, y1));
        assert!(game.player_alive == true);

        game.place_player(x2, y2);
        assert!(game.grid[x1 as usize][y1 as usize] == Block::None);
        assert!(game.grid[x2 as usize][y2 as usize] == Block::Player);
        assert!(game.player_pos == p(x2, y2));
        assert!(game.player_alive == true);
    }

    #[test]
    fn test_player_dies_when_falling_off_screen() {
        let mut game = Game::new(30, 30);
        game.place_player(15, 0);
        game.player_acceleration_from_gravity = 1.0;
        game.player_remaining_coyote_frames = 0;
        game.tick_physics();
        assert!(game.player_alive == false);
    }

    #[test]
    fn test_movecast() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);

        assert!(
            game.movecast(p(15, 9), p(15, 11))
                == Some(MovecastCollision {
                    pos: point![15, 9],
                    normal: vector![0, -1]
                })
        );
        assert!(
            game.movecast(p(15, 10), p(15, 11))
                == Some(MovecastCollision {
                    pos: point![15, 10],
                    normal: vector![0, 0]
                })
        );
        assert!(
            game.movecast(p(15, 9), p(17, 11))
                == Some(MovecastCollision {
                    pos: point![15, 9],
                    normal: vector![0, -1]
                })
        );
        assert!(
            game.movecast(p(15, 9), p(17, 110))
                == Some(MovecastCollision {
                    pos: point![15, 9],
                    normal: vector![0, -1]
                })
        );
        assert!(game.movecast(p(150, 9), p(17, 11)) == None);
        assert!(game.movecast(p(1, 9), p(-17, 9)) == None);
        assert!(game.movecast(p(15, 9), p(17, -11)) == None);
    }

    #[test]
    fn test_movecast_ignore_player() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);

        assert!(game.movecast(p(15, 11), p(15, 13)) == None);
    }
    #[test]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.in_world(p(0, 0)));
        assert!(game.in_world(p(29, 29)));
        assert!(!game.in_world(p(30, 30)));
        assert!(!game.in_world(p(10, -1)));
        assert!(!game.in_world(p(-1, 10)));
    }

    #[test]
    fn test_move_player() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_max_run_speed_bpf = 1.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(16, 11));
        assert!(game.grid[15][11] == Block::None);
        assert!(game.grid[16][11] == Block::Player);

        game.place_player(15, 11);
        assert!(game.player_desired_x_direction == 0);
        game.player_desired_x_direction = -1;

        game.tick_physics();
        game.tick_physics();

        assert!(game.grid[15][11] == Block::None);
        assert!(game.grid[13][11] == Block::Player);
        assert!(game.player_pos == p(13, 11));
    }
    #[test]
    fn test_stop_on_collision() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.draw_point((16, 11), Block::Wall);
        game.place_player(15, 11);
        game.player_max_run_speed_bpf = 1.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(15, 11));
        assert!(game.grid[16][11] == Block::Wall);
        assert!(game.grid[15][11] == Block::Player);
        assert!(game.player_vel_bpf.x == 0.0);
    }

    #[test]
    fn test_move_player_slowly() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_max_run_speed_bpf = 0.5;
        game.player_desired_x_direction = 1;

        game.tick_physics();

        assert!(game.player_pos == p(15, 11));

        game.tick_physics();

        assert!(game.player_pos == p(16, 11));
    }
    #[test]
    fn test_move_player_quickly() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);
        game.player_max_run_speed_bpf = 2.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();
        game.tick_physics();
        assert!(game.player_pos.x > 17);
        game.tick_physics();
        assert!(game.player_pos.x > 19);
    }
    #[test]
    fn test_fast_player_collision_between_frames() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        // Player should not teleport through this block
        game.draw_point((16, 11), Block::Wall);
        game.place_player(15, 11);
        game.player_max_run_speed_bpf = 2.0;
        game.player_desired_x_direction = 1;

        game.tick_physics();
        assert!(game.player_pos == p(15, 11));
        assert!(game.player_vel_bpf.x == 0.0);
    }
    #[test]
    fn test_can_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player_pos;

        game.player_jump();
        game.tick_physics();
        assert!(game.player_pos.y > start_pos.y);
    }
    #[test]
    fn test_player_gravity() {
        let mut game = Game::new(30, 30);
        game.place_player(15, 11);
        game.player_acceleration_from_gravity = 1.0;
        game.player_remaining_coyote_frames = 0;

        game.tick_physics();

        assert!(game.player_pos.y < 11);
    }

    #[test]
    fn test_slide_on_angled_collision() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.draw_line((14, 10), (14, 20), Block::Wall);

        game.place_player(15, 11);
        game.player_vel_bpf = v(-2.0, 2.0);
        game.tick_physics();
        assert!(game.player_vel_bpf.x == 0.0);
        assert!(game.player_vel_bpf.y > 0.0);
    }
    #[test]
    fn test_decellerate_when_already_moving_faster_than_max_speed() {
        let mut game = set_up_player_on_platform();
        game.player_max_run_speed_bpf = 1.0;
        game.player_vel_bpf.x = 5.0;
        game.player_desired_x_direction = 1;
        game.tick_physics();
        assert!(game.player_vel_bpf.x > game.player_max_run_speed_bpf);
        assert!(game.player_vel_bpf.x < 5.0);
    }
    #[test]
    fn test_no_double_jump() {
        let mut game = set_up_player_on_platform();
        game.player_jump_if_possible();
        game.tick_physics();
        let vel_y_before_second_jump = game.player_vel_bpf.y;
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player_vel_bpf.y < vel_y_before_second_jump);
    }

    #[test]
    fn test_respawn_button() {
        let mut game = Game::new(30, 30);
        game.handle_input(Event::Key(Key::Char('r')));
        assert!(game.player_alive);
        assert!(game.player_pos == p(15, 15));
    }

    #[test]
    fn test_wall_grab() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.tick_physics();
        assert!(game.player_vel_bpf.y == 0.0);
    }

    #[test]
    fn test_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player_vel_bpf.y > 0.0);
        assert!(game.player_vel_bpf.x > 0.0);
    }

    #[test]
    fn test_no_running_up_walls() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf = v(-1.0, 1.0);
        let start_vel_y = game.player_vel_bpf.y;
        // currently supported by coyote frames -> no gravity acceleration this tick
        // TODO: Should this happen?  Or should the player immediately start slowing down?
        game.tick_physics();
        // no longer supported by coyote frames due to vertical movement last tick
        game.tick_physics();
        assert!(game.player_vel_bpf.y < start_vel_y);
    }
    #[test]
    fn test_dont_grab_wall_while_moving_up() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf.y = 1.0;
        assert!(!game.player_is_grabbing_wall());
    }

    #[test]
    fn test_coyote_frames() {
        let mut game = Game::new(30, 30);
        game.place_player(15, 11);
        let start_pos = game.player_pos;
        game.player_acceleration_from_gravity = 1.0;
        game.player_max_coyote_frames = 1;
        game.player_remaining_coyote_frames = 1;
        game.tick_physics();
        assert!(game.player_pos.y == start_pos.y);
        game.tick_physics();
        assert!(game.player_pos.y < start_pos.y);
    }
    #[test]
    fn test_coyote_frames_dont_assist_jump() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_on_platform();

        game1.player_remaining_coyote_frames = 0;
        game1.player_jump_if_possible();
        game2.player_jump_if_possible();
        game1.tick_physics();
        game2.tick_physics();

        // Second tick after jump is where coyote physics may come into play
        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player_vel_bpf.y == game2.player_vel_bpf.y);
    }
    #[ignore]
    #[test]
    fn test_wall_jump_while_sliding_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_vel_bpf.y = 1.0;
        game.player_remaining_coyote_frames = 0;
        game.player_jump_if_possible();
        assert!(game.player_vel_bpf.x > 0.0);
    }

    #[ignore]
    #[test]
    // The general case of this is pressing jump any time before landing causing an instant jump when possible
    fn test_allow_early_jump() {}

    #[ignore]
    #[test]
    // The general case of this is allowing a single jump anytime while falling after walking (not jumping!) off a platform
    fn test_allow_late_jump() {}

    #[ignore]
    #[test]
    // This should allow high skill to lead to really fast wall climbing (like in N+)
    fn test_wall_jump_adds_velocity_instead_of_sets_it() {}

    #[test]
    fn test_draw_to_output_buffer() {
        let mut game = set_up_player_on_platform();
        game.update_output_buffer();
        assert!(game.get_buffered_glyph(game.player_pos) == EIGHTH_BLOCKS_FROM_LEFT[8]);
        assert!(game.get_buffered_glyph(game.player_pos + v(0, -1)) == Block::Wall.glyph());
    }

    #[test]
    fn test_horizontal_sub_glyph_positioning_on_right() {
        let mut game = set_up_player_on_platform();
        game.player_accumulated_pos_err.x = 0.5;
        game.update_output_buffer();
        assert!(game.get_buffered_glyph(game.player_pos) == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(game.get_buffered_glyph(game.player_pos + v(1, 0)) == EIGHTH_BLOCKS_FROM_LEFT[4]);
    }
    #[test]
    fn test_horizontal_sub_glyph_positioning_on_left() {
        let mut game = set_up_player_on_platform();
        game.player_accumulated_pos_err.x = -0.2;
        game.update_output_buffer();
        assert!(game.get_buffered_glyph(game.player_pos + v(-1, 0)) == EIGHTH_BLOCKS_FROM_LEFT[7]);
        assert!(game.get_buffered_glyph(game.player_pos) == EIGHTH_BLOCKS_FROM_LEFT[1]);
    }

    #[ignore]
    #[test]
    // At least when grabbing a wall, vertical subgrid positioning becomes a good idea
    fn test_sub_glyph_positioning_on_vertical() {}

    #[ignore]
    #[test]
    // Once we have vertical subsquare positioning up and running, a slow slide down will look cool.
    fn test_slowly_slide_down_when_grabbing_wall() {}
}
