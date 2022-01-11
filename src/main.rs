extern crate line_drawing;
extern crate std;
extern crate termion;

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

    fn does_fall(&self) -> bool {
        match self {
            Block::None | Block::Wall => false,
            _ => true,
        }
    }
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
    player_speed: (i32, i32),
    player_desired_direction: (i32, i32),
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
            player_speed: (0, 0),
            player_desired_direction: (0, 0),
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
            self.grid[self.player_pos.0 as usize][self.player_pos.1 as usize] = Block::None;
        }
        self.grid[x as usize][y as usize] = Block::Player;
        self.player_speed = (0, 0);
        self.player_desired_direction = (0, 0);
        self.player_pos = (x, y);
        self.player_alive = true;
    }

    // When The player presses the jump button
    fn player_jump(&mut self) {
        self.player_speed.1 = 3;
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
                Key::Char('a') | Key::Left => self.player_desired_direction.0 = -1,
                Key::Char('s') | Key::Down => self.player_desired_direction.0 = 0,
                Key::Char('d') | Key::Right => self.player_desired_direction.0 = 1,
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
        self.apply_player_motion();
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
                if block.does_fall() {
                    let is_bottom_row = y == 0;
                    let has_direct_support = !is_bottom_row && self.grid[x][y - 1] != Block::None;
                    if is_bottom_row {
                        self.grid[x][y] = Block::None;
                        if block == Block::Player {
                            self.player_alive = false;
                        }
                    } else if !has_direct_support {
                        self.grid[x][y - 1] = block;
                        self.grid[x][y] = Block::None;
                        if block == Block::Player {
                            self.player_pos.1 -= 1;
                        }
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

    // not counting gravity
    fn apply_player_motion(&mut self) {
        if self.player_desired_direction.0 != 0 && self.player_is_supported() {
            let target_x = self.player_pos.0 + self.player_desired_direction.0;
            let target_y = self.player_pos.1;
            if self.in_world(target_x, target_y)
                && self.grid[target_x as usize][target_y as usize] == Block::None
            {
                self.move_player_to(target_x, target_y);
            }
        }
    }

    fn player_is_supported(&self) -> bool {
        self.player_pos.1 > 0
            && self.grid[self.player_pos.0 as usize][self.player_pos.1 as usize] != Block::None
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

    while game.running {
        while let Ok(evt) = rx.try_recv() {
            game.handle_input(evt);
        }
        game.tick_physics();
        game.update_screen(&mut stdout);
        thread::sleep(Duration::from_millis(20));
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
    fn place_player() {
        let mut game = Game::new(30, 30);
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
    fn move_player() {
        let mut game = Game::new(30, 30);
        game.draw_line((10, 10), (20, 10), Block::Wall);
        game.place_player(15, 11);

        game.player_desired_direction = (1, 0);

        game.tick_physics();

        assert_eq!(game.grid[15][11], Block::None);
        assert_eq!(game.grid[16][11], Block::Player);
        assert_eq!(game.player_pos, (16, 11));
        
        game.place_player(15, 11);
        assert_eq!(game.player_desired_direction, (0, 0));
    }
}
