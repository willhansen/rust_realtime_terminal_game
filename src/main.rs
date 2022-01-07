extern crate line_drawing;
extern crate std;
extern crate termion;

use std::io::{stdin, stdout, Write};
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

struct Game {
    grid: Vec<Vec<char>>,
    stdout: MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
}

impl Game {
    // fn set_output<W: std::io::Write>(&mut self, stdout: &mut MouseTerminal<W>) {
    //     self.stdout = stdout;
    // }
    // 
    fn new_game() -> Game {
        let (width, height) = termion::terminal_size().unwrap();
        Game {
            grid: vec![vec![' '; height as usize]; width as usize],
            stdout: MouseTerminal::from(stdout().into_raw_mode().unwrap()),
        }
    }

    fn draw_line(
        &mut self,
        pos0: (u16, u16),
        pos1: (u16, u16),
        character: char,
    ) {
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

    fn draw_point(
        &mut self,
        pos: (u16, u16),
        character: char,
    ) {
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
}

fn main() {
    let stdin = stdin();
    let mut prev_mouse_pos = (1, 1);
    let selected_char = '█';
    let mut game = Game::new_game();

    write!(
        game.stdout,
        "{}{}q to exit. Click, click, click!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    game.stdout.flush().unwrap();

    for c in stdin.events() {
        let evt = c.unwrap();
        match evt {
            Event::Key(Key::Char('q')) => break,
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
}
