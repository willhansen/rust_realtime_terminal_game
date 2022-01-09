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
    stdout: MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    prev_mouse_pos: (u16, u16), // where mouse was last frame (if pressed)
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
            stdout: MouseTerminal::from(stdout().into_raw_mode().unwrap()),
            prev_mouse_pos: (1, 1),
            last_pressed_key: None,
            running: true,
            selected_block: Block::Wall,
            player_pos: (0, 0),
            player_speed: (0, 0),
            player_alive: false,
        }
    }

    fn draw_line(&mut self, pos0: (u16, u16), pos1: (u16, u16), block: Block) {
        for (x1, y1) in line_drawing::Bresenham::new(
            (pos0.0 as i32, pos0.1 as i32),
            (pos1.0 as i32, pos1.1 as i32),
        ) {
            self.grid[x1 as usize][y1 as usize] = block;
            write!(
                self.stdout,
                "{}{}",
                termion::cursor::Goto(x1 as u16 + 1, y1 as u16 + 1),
                block.glyph()
            )
            .unwrap();
        }
    }

    fn draw_point(&mut self, pos: (u16, u16), block: Block) {
        self.grid[pos.0 as usize][pos.1 as usize] = block;
        write!(
            self.stdout,
            "{}{}",
            termion::cursor::Goto(pos.0 + 1, pos.1 + 1),
            block.glyph()
        )
        .unwrap();
    }

    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![Block::None; height as usize]; width as usize];
        write!(
            self.stdout,
            "{}{}",
            termion::clear::All,
            termion::cursor::Goto(1, 1)
        )
        .unwrap();
    }
    fn place_player(&mut self, x: u16, y: u16) {
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
                MouseEvent::Press(MouseButton::Left, x_from_1, y_from_1) => {
                    let (x, y) = (x_from_1 - 1, y_from_1 - 1);
                    self.draw_point((x, y), self.selected_block);
                    write!(self.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Press(MouseButton::Right, x_from_1, y_from_1) => {
                    let (x, y) = (x_from_1 - 1, y_from_1 - 1);
                    self.place_player(x, y);
                    self.draw_point((x, y), Block::Player);
                    write!(self.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                }
                MouseEvent::Hold(x_from_1, y_from_1) => {
                    let (x, y) = (x_from_1 - 1, y_from_1 - 1);
                    self.draw_line(self.prev_mouse_pos, (x, y), self.selected_block);
                    write!(self.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    self.prev_mouse_pos = (x, y);
                }
                _ => {}
            },
            _ => {}
        }
        self.stdout.flush().unwrap();
    }

    fn tick_physics(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        let old_grid = self.grid.clone();

        self.apply_gravity();
        // self.apply_player_motion();

        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if self.grid[x][y] != old_grid[x][y] {
                    write!(
                        self.stdout,
                        "{}{}",
                        termion::cursor::Goto(x as u16, y as u16),
                        self.grid[x][y].glyph(),
                    )
                    .unwrap();
                }
            }
        }
        write!(self.stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        self.stdout.flush().unwrap();
    }

    fn apply_gravity(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for forward_y in 0..height {
                // We want to count from high y to low y, so things fall correctly
                let y = (height - 1) - forward_y;
                let block = self.grid[x][y];
                if block.does_fall() {
                    let is_bottom_row = y == (height - 1);
                    let has_direct_support = !is_bottom_row && self.grid[x][y + 1] != Block::None;
                    if is_bottom_row {
                        self.grid[x][y] = Block::None;
                        if block == Block::Player {
                            self.player_alive = false;
                        }
                    } else if !has_direct_support {
                        self.grid[x][y + 1] = block;
                        self.grid[x][y] = Block::None;
                        if block == Block::Player {
                            self.player_pos.1 += 1;
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
        thread::sleep(Duration::from_millis(20));
    }
}
