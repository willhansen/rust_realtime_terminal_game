extern crate line_drawing;
extern crate termion;

use std::io::{stdin, stdout, Write};
use termion::event::{Event, Key, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

fn main() {
    let stdin = stdin();
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());
    let mut prev_mouse_pos = (1, 1);

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
                write!(stdout, "{}{}", termion::clear::All, termion::cursor::Goto(1, 1)).unwrap();
            }
            Event::Mouse(me) => match me {
                MouseEvent::Press(_, x, y) => {
                    draw_point(&mut stdout, (x, y));
                    write!(stdout, "{}", termion::cursor::Goto(1, 1)).unwrap();
                    prev_mouse_pos = (x, y);
                }
                MouseEvent::Hold(x, y) => {
                    draw_line(&mut stdout, prev_mouse_pos, (x, y));
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

fn draw_point<W: std::io::Write>(stdout: &mut MouseTerminal<W>, pos: (u16, u16)) {
    write!(stdout, "{}x", termion::cursor::Goto(pos.0, pos.1)).unwrap();
}

fn draw_line<W: std::io::Write>(stdout: &mut MouseTerminal<W>, pos0: (u16, u16), pos1: (u16, u16)) {
    for (x1, y1) in line_drawing::Bresenham::new(
        (pos0.0 as i32, pos0.1 as i32),
        (pos1.0 as i32, pos1.1 as i32),
    ) {
        write!(stdout, "{}x", termion::cursor::Goto(x1 as u16, y1 as u16)).unwrap();
    }
}
