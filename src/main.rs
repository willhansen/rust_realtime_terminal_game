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
            Block::Brick => '▓',
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
}

struct Game {
    grid: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    stdout: MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    prev_mouse_pos: (u16, u16), // where mouse was last frame (if pressed)
    running: bool,              // set false to quit
    selected_block: Block,      // What the mouse places
}

impl Game {
    fn new_game() -> Game {
        let (width, height) = termion::terminal_size().unwrap();
        Game {
            grid: vec![vec![Block::None; height as usize]; width as usize],
            stdout: MouseTerminal::from(stdout().into_raw_mode().unwrap()),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
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
                termion::cursor::Goto(x1 as u16, y1 as u16),
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
            termion::cursor::Goto(pos.0, pos.1),
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
    fn handle_input(&mut self, evt: termion::event::Event) {
        match evt {
            Event::Key(Key::Char('q')) => {
                self.running = false;
            }
            // 'c' to clear screen
            Event::Key(Key::Char('c')) => {
                self.clear();
            }
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, x, y) => {
                    self.draw_point((x, y), self.selected_block);
                    write!(self.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Hold(x, y) => {
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
        let mut new_grid = vec![vec![Block::None; height as usize]; width as usize];

        for x in 0..width {
            for forward_y in 0..height {
                // We want to count from high y to low y, so things fall correctly
                let y = height - forward_y - 1;
                let block = &self.grid[x][y];
                // EVERYTHING falls
                if !matches!(*block, Block::None) {
                    if y < height - 1 {
                        new_grid[x][y + 1] = self.grid[x][y];
                    }
                    // Don't actually need this
                    // new_grid[x][y] = ' ';
                }
            }
        }

        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if &self.grid[x][y] != &new_grid[x][y] {
                    write!(
                        self.stdout,
                        "{}{}",
                        termion::cursor::Goto(x as u16, y as u16),
                        new_grid[x][y].glyph(),
                    )
                    .unwrap();
                }
            }
        }
        write!(self.stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        self.stdout.flush().unwrap();
        self.grid = new_grid;
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
