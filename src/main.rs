extern crate line_drawing;
extern crate std;
extern crate termion;

use std::io::{stdin, stdout, Write};
use std::process;
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

struct Game {
    grid: Vec<Vec<char>>,
    stdout: MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
}

impl Game {
    fn new_game() -> Game {
        let (width, height) = termion::terminal_size().unwrap();
        Game {
            grid: vec![vec![' '; height as usize]; width as usize],
            stdout: MouseTerminal::from(stdout().into_raw_mode().unwrap()),
        }
    }

    fn draw_line(&mut self, pos0: (u16, u16), pos1: (u16, u16), character: char) {
        for (x1, y1) in line_drawing::Bresenham::new(
            (pos0.0 as i32, pos0.1 as i32),
            (pos1.0 as i32, pos1.1 as i32),
        ) {
            self.grid[x1 as usize][y1 as usize] = character;
            write!(
                self.stdout,
                "{}{}",
                termion::cursor::Goto(x1 as u16, y1 as u16),
                character
            )
            .unwrap();
        }
    }

    fn draw_point(&mut self, pos: (u16, u16), character: char) {
        self.grid[pos.0 as usize][pos.1 as usize] = character;
        write!(
            self.stdout,
            "{}{}",
            termion::cursor::Goto(pos.0, pos.1),
            character
        )
        .unwrap();
    }

    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![' '; height as usize]; width as usize];
        write!(
            self.stdout,
            "{}{}",
            termion::clear::All,
            termion::cursor::Goto(1, 1)
        )
        .unwrap();
    }

    fn tick_physics(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        let mut new_grid = vec![vec![' '; height as usize]; width as usize];

        for x in 0..width {
            for forward_y in 0..height {
                // We want to count from high y to low y, so things fall correctly
                let y = height - forward_y - 1;
                let character = &self.grid[x][y];
                // EVERYTHING falls
                if *character != ' ' {
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
                if self.grid[x][y] != new_grid[x][y] {
                    write!(
                        self.stdout,
                        "{}{}",
                        termion::cursor::Goto(x as u16, y as u16),
                        new_grid[x][y],
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
    let mut prev_mouse_pos = (1, 1);
    let selected_char = '█';
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

    'mainloop: loop {
        while let Ok(evt) = rx.try_recv() {
            match evt {
                Event::Key(Key::Char('q')) => break 'mainloop,
                // 'c' to clear screen
                Event::Key(Key::Char('c')) => {
                    game.clear();
                }
                Event::Mouse(me) => match me {
                    MouseEvent::Press(MouseButton::Left, x, y) => {
                        game.draw_point((x, y), selected_char);
                        write!(game.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                        prev_mouse_pos = (x, y);
                    }
                    MouseEvent::Hold(x, y) => {
                        game.draw_line(prev_mouse_pos, (x, y), selected_char);
                        write!(game.stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                        prev_mouse_pos = (x, y);
                    }
                    _ => {}
                },
                _ => {}
            }
            game.stdout.flush().unwrap();
        }
        game.tick_physics();
        thread::sleep(Duration::from_millis(20));
    }
}
