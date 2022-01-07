extern crate line_drawing;
extern crate std;
extern crate termion;

use std::io::{stdin, stdout, Write};
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

struct Game {
    grid: Vec<Vec<char>>,
}
impl Game {
    fn draw_line<W: std::io::Write>(
        &mut self,
        stdout: &mut MouseTerminal<W>,
        pos0: (u16, u16),
        pos1: (u16, u16),
        character: char,
    ) {
        for (x1, y1) in line_drawing::Bresenham::new(
            (pos0.0 as i32, pos0.1 as i32),
            (pos1.0 as i32, pos1.1 as i32),
        ) {
            write!(
                stdout,
                "{}{}",
                termion::cursor::Goto(x1 as u16, y1 as u16),
                character
            )
            .unwrap();
        }
    }

    fn draw_point<W: std::io::Write>(
        &mut self,
        stdout: &mut MouseTerminal<W>,
        pos: (u16, u16),
        character: char,
    ) {
        write!(
            stdout,
            "{}{}",
            termion::cursor::Goto(pos.0, pos.1),
            character
        )
        .unwrap();
    }
}

fn main() {
    let stdin = stdin();
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());
    let mut prev_mouse_pos = (1, 1);
    let (width, height) = termion::terminal_size().unwrap();
    let selected_char = '█';
    let mut game = Game {
        grid: vec![vec![' '; width as usize]; height as usize],
    };

    write!(
        stdout,
        "{}{}q to exit. Click, click, click!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    stdout.flush().unwrap();

    for c in stdin.events() {
        let evt = c.unwrap();
        match evt {
            Event::Key(Key::Char('q')) => break,
            Event::Key(Key::Char('c')) => {
                write!(
                    stdout,
                    "{}{}",
                    termion::clear::All,
                    termion::cursor::Goto(1, 1)
                )
                .unwrap();
            }
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, x, y) => {
                    game.draw_point(&mut stdout, (x, y), selected_char);
                    write!(stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    prev_mouse_pos = (x, y);
                }
                MouseEvent::Hold(x, y) => {
                    game.draw_line(&mut stdout, prev_mouse_pos, (x, y), selected_char);
                    write!(stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    prev_mouse_pos = (x, y);
                }
                _ => {}
            },
            _ => {}
        }
        stdout.flush().unwrap();
    }
}
