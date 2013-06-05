from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import NumericProperty, ListProperty, ObjectProperty, \
        BooleanProperty
from kivy.utils import get_color_from_hex
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.animation import Animation
from functools import partial
from random import randint, random

SIZE = 8

COLORS = map(get_color_from_hex, (
    '#95a5a6', # white
    '#c0392b', # red
    '#d35400', # orange
    '#8e44ad', # purple
    '#f1c40f', # yellow
    '#3498db', # blue
    '#2ecc71', # green
    ))

Builder.load_string('''
<Jewel>:
    canvas:
        Color:
            rgba: 236 / 255., 240 / 255., 241 / 255., int(self.selected)
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgb: root.color
        Rectangle:
            pos: self.x + 5, self.y + 5
            size: self.width - 10, self.height - 10

<Board>:
    canvas:
        Color:
            rgb: .1, .1, .1
        Rectangle:
            pos: self.pos
            size: self.size
''')


class Jewel(Widget):
    color = ListProperty([0, 0, 0])
    index = NumericProperty(0)
    ix = NumericProperty(0)
    iy = NumericProperty(0)
    board = ObjectProperty()
    selected = BooleanProperty(False)
    animating = BooleanProperty(False)
    anim = ObjectProperty(None, allownone=True)

    def animate_to(self, x, y, d=0):
        self.animating = True
        if self.anim:
            self.anim.cancel(self)
            self.anim = None

        #distance = Vector(*self.pos).distance(Vector(x, y))
        #distance /= float(self.board.jewel_size)
        #duration = distance * .1
        duration = .20

        anim = Animation(pos=(x, y), d=duration, t='out_sine')
        if d:
            anim = Animation(d=d) + anim
        anim.bind(on_complete=self.on_complete)
        self.anim = anim
        anim.start(self)

    def on_complete(self, *args):
        self.animating = False
        self.board.check(self)

    def stop(self):
        if self.anim:
            self.anim.cancel(self)
            self.anim = None

    def explode(self, *args):
        anim = Animation(pos=(self.x - 20, self.y - 20),
                  size=(self.width + 40, self.height + 40),
                  d=.2)
        anim.bind(on_complete=self.destroy)
        anim.start(self)

    def destroy(self, *args):
        self.board.remove_widget(self)


class SquareLayout(FloatLayout):
    def do_layout(self, *args):
        s = self.width
        if self.width > self.height:
            s = self.height
        for child in self.children:
            child.size = s, s
            child.center = self.center


class Board(Widget):

    def __init__(self, **kwargs):
        super(Board, self).__init__(**kwargs)

        # initalize the board
        self.board = []
        for index in range(SIZE):
            self.board += [[None] * SIZE]

        # fill the board
        Clock.schedule_once(self.fill_board, 1)

    def fill_board(self, *args):
        for ix in range(SIZE):
            for iy in range(SIZE):
                jewel = self.board[ix][iy]
                if jewel:
                    continue
                jewel = self.generate()
                jewel.ix = ix
                jewel.iy = iy
                x, y = self.index_to_pos(ix, iy)
                ax, ay = self.index_to_pos(ix, iy + SIZE)
                jewel.pos = ax, ay
                jewel.animate_to(x, y, d=iy / 10. + random() / 10.)
                self.board[ix][iy] = jewel


    def index_to_pos(self, ix, iy):
        js = self.jewel_size
        return self.x + ix * js, self.y + iy * js

    def touch_to_index(self, tx, ty):
        tx -= self.x
        ty -= self.y
        js = self.jewel_size
        return int(tx / js), int(ty / js)

    @property
    def jewel_size(self):
        return int(self.width / SIZE)

    def generate(self):
        index = randint(0, 6)
        color = COLORS[index]
        js = self.jewel_size
        jewel = Jewel(index=index, color=color, board=self,
                size=(js, js))
        self.add_widget(jewel)
        return jewel


    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return
        touch.grab(self)
        ix, iy = self.touch_to_index(*touch.pos)
        jewel = self.board[ix][iy]
        jewel.selected = True

        touch.ud['source'] = ix, iy
        touch.ud['jewel'] = jewel
        touch.ud['action'] = False

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return
        self.check_touch_swap(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)
        jewel = touch.ud['jewel']
        jewel.selected = False
        self.check_touch_swap(touch)

    def check_touch_swap(self, touch):
        if touch.ud['action']:
            return
        ix, iy = self.touch_to_index(*touch.pos)
        sx, sy = touch.ud['source']

        # index out of bounds ?
        if ix < 0 or ix >= SIZE or iy < 0 or iy >= SIZE:
            return
        # check if we moved
        if ix == iy and sx == sy:
            return

        # we moved, but ensure it's only from "1"
        dx = ix - sx
        dy = iy - sy
        if dx == 0 and dy == 0:
            return

        # try to swap
        ix, iy = ix2, iy2 = touch.ud['source']
        if abs(dx) > abs(dy):
            ix2 += 1 if dx > 0 else -1
        else:
            iy2 += 1 if dy > 0 else -1

        self.swap(ix, iy, ix2, iy2)
        touch.ud['action'] = True

    def swap(self, ix1, iy1, ix2, iy2):
        jewel1 = self.board[ix1][iy1]
        jewel2 = self.board[ix2][iy2]
        if jewel1 is None or jewel2 is None:
            return
        self.board[ix1][iy1] = jewel2
        self.board[ix2][iy2] = jewel1
        jewel1.ix, jewel1.iy = ix2, iy2
        jewel2.ix, jewel2.iy = ix1, iy1

        jewel1.animate_to(*self.index_to_pos(ix2, iy2))
        jewel2.animate_to(*self.index_to_pos(ix1, iy1))

    def check(self, jewel):
        sel_all = []
        sel_x = []
        sel_y = []
        board = self.board
        index = jewel.index
        ix, iy = jewel.ix, jewel.iy

        for x in xrange(ix - 1, -1, -1):
            j = board[x][iy]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_x.append(j)

        for x in xrange(ix + 1, SIZE):
            j = board[x][iy]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_x.append(j)

        for y in xrange(iy - 1, -1, -1):
            j = board[ix][y]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_y.append(j)

        for y in xrange(iy + 1, SIZE):
            j = board[ix][y]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_y.append(j)

        if len(sel_all) < 2:
            return
        if len(sel_x) < 2 and len(sel_y) < 2:
            return

        if len(sel_x) >= 2 and len(sel_y) >= 2:
            self.bam(jewel, *sel_all)
        elif len(sel_x) >= 2:
            self.bam(jewel, *sel_x)
        elif len(sel_y) >= 2:
            self.bam(jewel, *sel_y)

    def bam(self, *jewels):
        # first explode all the jewel
        d = 0
        board = self.board
        for jewel in jewels:
            ix, iy = jewel.ix, jewel.iy
            board[ix][iy] = None
            Clock.schedule_once(jewel.explode, d)
            d += .2
        Clock.schedule_once(partial(self.bam_2, jewels), d)

    def bam_2(self, jewels, *args):
        # then make them fall down
        board = self.board

        # fall down
        for ix in xrange(SIZE):
            missing = board[ix].count(None)
            if missing == 0:
                continue

            row_jewels = [x for x in board[ix] if x is not None]
            board[ix] = [None] * SIZE

            iy = 0
            for jewel in row_jewels:
                if jewel.iy != iy:
                    jewel.iy = iy
                    jewel.animate_to(*self.index_to_pos(ix, iy))
                board[ix][iy] = jewel
                iy += 1

            for iy in xrange(iy, SIZE):
                jewel = self.generate()
                jewel.ix = ix
                jewel.iy = iy
                x, y = self.index_to_pos(ix, iy)
                ax, ay = self.index_to_pos(ix, iy + missing)
                jewel.pos = ax, ay
                jewel.animate_to(x, y)#, d=(iy - missing) / 10.)
                board[ix][iy] = jewel


class JewelApp(App):
    def build(self):
        root = SquareLayout()
        root.add_widget(Board())
        return root

JewelApp().run()
