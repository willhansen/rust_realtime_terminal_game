extern crate line_drawing;
extern crate std;
extern crate termion;

use std::collections::HashSet;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;
use termion::color;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq)]
enum Block {
    None,
    Wall,
    Brick,
    Sand,
    Water,
    Player,
}

impl Block {
    fn glyph(&self) -> char {
        match self {
            Block::None => ' ',
            Block::Wall => '█',
            Block::Brick => '▪',
            Block::Water => '█',
            Block::Player => '🯅',
            _ => 'E',
        }
    }

    fn color_str(&self) -> String {
        match self {
            Block::None => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
                color::Bg(color::Black).to_string()
            ),
            Block::Wall => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
                color::Bg(color::Black).to_string()
            ),
            Block::Brick => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
                color::Bg(color::Black).to_string()
            ),
            Block::Water => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
                color::Bg(color::Black).to_string()
            ),
            Block::Player => format!(
                "{}{}",
                color::Fg(color::White).to_string(),
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
    grid: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    prev_grid: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    stdout: MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    terminal_size: (u16, u16),  // (width, height)
    prev_mouse_pos: (i32, i32), // where mouse was last frame (if pressed)
    last_pressed_key: Option<termion::event::Key>,
    running: bool,         // set false to quit
    selected_block: Block, // What the mouse places
    player_alive: bool,
    player_pos: (i32, i32),
    player_speed: (i32, i32),
}

impl Game {
    fn new_game() -> Game {
        let (width, height) = termion::terminal_size().unwrap();
        Game {
            grid: vec![vec![Block::None; height as usize]; width as usize],
            prev_grid: vec![vec![Block::None; height as usize]; width as usize],
            stdout: MouseTerminal::from(stdout().into_raw_mode().unwrap()),
            terminal_size: termion::terminal_size().unwrap(),
            prev_mouse_pos: (1, 1),
            last_pressed_key: None,
            running: true,
            selected_block: Block::Wall,
            player_pos: (0, 0),
            player_speed: (0, 0),
            player_alive: false,
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
        self.player_pos = (0, 0);
        self.player_alive = true;
    }

    // When The player presses the jump button
    fn player_jump(&mut self) {
        self.player_speed.1 = 3;
    }

    // When the player presses a horizontal movement button (or down, for stop)
    // x direction only.  positive is right.  zero is stopped
    fn player_move(&mut self, x: i32) {
        self.player_speed.0 = x;
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
                Key::Char('a') | Key::Left => self.player_move(-1),
                Key::Char('s') | Key::Down => self.player_move(0),
                Key::Char('d') | Key::Right => self.player_move(1),
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
        self.stdout.flush().unwrap();
    }

    fn tick_physics(&mut self) {
        self.apply_gravity();
        // self.apply_player_motion();
    }
    
    fn update_screen(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if self.grid[x][y] != self.prev_grid[x][y] {
                    let (term_x, term_y) = self.world_to_screen(&(x as i32, y as i32));
                    write!(
                        self.stdout,
                        "{}{}",
                        termion::cursor::Goto(term_x, term_y),
                        self.grid[x][y].glyph(),
                    )
                    .unwrap();
                }
            }
        }
        write!(self.stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        self.stdout.flush().unwrap();
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
}

fn main() {
    let stdin = stdin();
    let mut game = Game::new_game();

    write!(
        game.stdout,
        "{}{}q to exit.  c to clear.  Mouse to draw.  Begin!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    game.stdout.flush().unwrap();
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
        game.update_screen();
        thread::sleep(Duration::from_millis(20));
    }
}
