import logging
import curses
import sys
from curses import wrapper
import signal


class CursesHandler(logging.Handler):
    def __init__(self, screen):
        logging.Handler.__init__(self)
        self.screen = screen

    def emit(self, record):
        try:
            msg = self.format(record)
            screen = self.screen
            fs = "\n%s"
            screen.addstr(fs % msg)
            screen.box()
            screen.refresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise

    def write(self, msg):
        try:
            screen = self.screen
            screen.addstr('  ' + msg)
            screen.box()
            screen.refresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise


def draw_console(screen):
    num_rows, num_cols = screen.getmaxyx()
    console_pos = num_rows * 2 // 3

    win = screen.subwin(console_pos, 0)
    win.box()
    win.addstr(1, 2, "Console")
    win1 = win.subwin(console_pos, 0)

    win1.refresh()
    MAX_ROW, MAX_COL = win1.getmaxyx()
    win1.scrollok(True)
    win1.idlok(True)
    win1.leaveok(True)
    win1.setscrreg(3, MAX_ROW - 2)
    win1.addstr(4, 4, "")

    mh = CursesHandler(win1)
    win1.move(3, 0)
    formatter = logging.Formatter(' %(asctime) -25s - %(name) -15s - %(levelname) -10s - %(message)s')
    formatterDisplay = logging.Formatter('  %(asctime)-8s|%(name)-12s|%(levelname)-6s|%(message)-s', '%H:%M:%S')
    mh.setFormatter(formatterDisplay)
    logger = logging.getLogger('gui-vispipe')
    logger.addHandler(mh)
    return mh


def MainWindow(screen):
    curses.curs_set(0)
    logger = draw_console(screen)

    win = screen.subwin(0, 0)
    win.box()
    win.addstr(1, 2, "Queues State")
    win.refresh()
    return screen, win, logger


def draw_bar(value, max_value, size=10):
    ratio = value / max_value
    n = int(size * ratio)
    return ''.join(['|' * n, '-' * (size - n)])


class CursesQueueGUI:
    def __init__(self):
        self.screen, self.queue_win, self.logger = wrapper(MainWindow)
        signal.signal(signal.SIGINT, self.signal_handler)
        sys.stdout = self.logger
        sys.stderr = self.logger

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        self.queues = {}
        self.size = 20

    def signal_handler(self, sig, frame):
        print('[ Press Enter to exit ]')
        self.screen.getch()
        curses.endwin()
        sys.exit(0)

    def refresh(self):
        self.queue_win.refresh()
        self.queue_win.box()
        curses.napms(100)

    def set_queues(self, hash, name, values, max_values):
        self.queues.setdefault(hash, []).append((name, values, max_values))
        self.draw_queues()
        self.refresh()

    def draw_queues(self):
        row = 2
        column = 3
        MAX_ROW, MAX_COL = self.queue_win.getmaxyx()
        for k in self.queues.keys():
            for qith in self.queues[k]:
                # Clear the line
                self.queue_win.addstr(row, column, ' ' * (self.size + 2))

                self.queue_win.addstr(row, column, '%s %s' % (k, qith[0]))
                bar = draw_bar(qith[1], qith[2], size=self.size)
                color = curses.color_pair(2)
                if qith[1] > 1 / 2 * qith[2]:
                    color = curses.color_pair(1)

                # Clear those lines
                self.queue_win.addstr(row + 1, column, ' ' * (self.size + 2))

                # Draw actual content
                self.queue_win.addstr(row + 1, column, '[', curses.A_BOLD)
                self.queue_win.addstr(row + 1, column + 1, '%s' % bar, color)
                self.queue_win.addstr(row + 1, column + len(bar), ']', curses.A_BOLD)
                row += 2
                if row % (MAX_ROW * 2 // 3 - 1) in [0, 1]:
                    row = 2
                    column += self.size + 5
                if column > MAX_COL - self.size:
                    return

    def clear_queues(self):
        self.queues = {}
        self.refresh()


if __name__ == "__main__":
    gui = CursesQueueGUI()
    logging = logging.getLogger(__name__)

    while True:
        import numpy as np
        x = np.random.randint(0, 100)
        logging.error('Logging the fake queue %s' % x)
        gui.set_queues(hash(gui), 'test_random', x, x + np.random.randint(0, 500))
