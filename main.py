'''
Flat Jewels
===========

.. author: Mathieu Virbel

You can play to the game freely.
You are not allowed to distribute your own game based on this code, modified or
not.

Credits
-------

* Tetra_-_Hunters: https://www.jamendo.com/fr/track/656920/hunters
* Jewel movement: http://www.freesound.org/people/junggle/sounds/28826/
* Warning (modified): http://www.freesound.org/people/Ultranova105/sounds/136756/
* Time 5 (modified): http://www.freesound.org/people/fins/sounds/133427/
* Time 10 (modified): http://www.freesound.org/people/fins/sounds/133429/
* End (modified): http://www.freesound.org/people/fins/sounds/171670/
* Levelup (modified): http://www.freesound.org/people/JustinBW/sounds/80921/

'''

__version__ = '0.4.1'

import json
from os.path import exists, join
#from kivy.config import Config
#Config.set('graphics', 'maxfps', '30')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import NumericProperty, ListProperty, ObjectProperty, \
        BooleanProperty, DictProperty
from kivy.utils import get_color_from_hex, boundary
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.core.image import Image as CoreImage
from kivy.uix.label import Label
from kivy.metrics import sp
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.core.audio import SoundLoader
from functools import partial
from random import randint, random
from math import sin
from time import time

HIGHSCORE_VERSION = 1
PLATFORM = platform()

if PLATFORM == 'android':
    import android

SIZE = 8
LEVEL_TIME = 60
HINT_TIME = 4

# Percentage of timed jewel to generate
PERCENT_TIMED_JEWEL = 14 / 100.
# Decrease of the percentage after each level
DECREASE_TIMED_JEWEL = 0.95
# Percentage of big timed jewel to generate (10 instead of 5)
PERCENT_BIGTIME_JEWEL = 20 / 100.

if 0:
    LEVEL_TIME = 10
    PERCENT_TIMED_JEWEL = 1.

COLORS = [get_color_from_hex(x) for x in (
    '#95a5a6', # white
    '#de3c3c', # red
    '#e3710e', # orange
    '#b155d8', # purple
    '#f1c40f', # yellow
    '#3498db', # blue
    '#2ecc71', # green
    )]

BACKGROUND_COLORS = [get_color_from_hex(x) for x in (
    '0dc1f2', # light blue
    'f25dee', # light pink
    'f1f20d', # light yellow
    '69c005', # dark green
    '631db9', # dark purple
    'cb0505', # dark red
)]

PLATFORM = platform()
if PLATFORM not in ('android', 'ios'):
    Builder.load_string('''
<Label>:
    font_name: 'data/coolvetica rg.ttf'
''')

Builder.load_string('''
<ScoreLabel>:
    font_size: '30sp'
    size_hint: None, None
    center: self.origin[0], self.origin[1] + self.dy

<TitleText>:
    font_size: '50sp'

<GameOverGraphBar>:
    orientation: 'vertical'

    Widget:
        size_hint_y: None
        height: '20sp'

    Widget:
        id: wd
        canvas:
            Color:
                rgb: app.background_rgb
            Rectangle:
                pos: self.pos
                size: self.width, self.height * root.alphascore

        Label:
            id: wdl
            text: root.to_human(root.score * root.alpha)
            font_size: '20sp'
            center_x: wd.center_x
            y: wd.y + wd.height * root.alphascore
            height: self.texture_size[1]


    Label:
        text: '{}'.format(root.level)
        font_size: '20sp'
        text_size: (self.width, None)
        halign: 'center'
        height: self.texture_size[1]
        size_hint_y: None

<JewelButton@Button>:
    background_down: 'data/gem_selected.png'
    background_normal: 'data/gem.png'
    border: (32, 32, 32, 32)

<Jewel>:
    canvas:
        Color:
            rgb: root.color
        Rectangle:
            pos: self.x + 5, self.y + 5
            size: self.width - 10, self.height - 10
            texture: app.textures['gem_selected' if self.selected else 'gem']

<AreaBombJewel>:
    canvas:
        Color:
            rgb: 1, 1, 1
        Rectangle:
            pos: self.x + 5, self.y + 5
            size: self.width - 10, self.height - 10
            texture: app.textures['tarea']

<LineBombJewel>:
    canvas:
        Color:
            rgb: 1, 1, 1
        Rectangle:
            pos: self.x + 5, self.y + 5
            size: self.width - 10, self.height - 10
            texture: app.textures['tline']

<TimedJewel>:
    canvas:
        Color:
            rgb: 1, 1, 1
        Rectangle:
            pos: self.x + 5, self.y + 5
            size: self.width - 10, self.height - 10
            texture: app.textures['t{}'.format(self.time)]

<-SmoothLabel@Label>:
    font_name: 'data/coolvetica rg.ttf'
    canvas:
        Color:
            rgb: 1, 1, 1
        Rectangle:
            pos: self.center_x - self.texture_size[0] / 2., self.center_y - self.texture_size[1] / 2.
            size: self.texture_size
            texture: self.texture

<Timer@Widget>:
    canvas:
        Color:
            rgb: app.background_rgb
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgb: 1, 1 - app.sidebar_warning_color[-1], 1 - app.sidebar_warning_color[-1]
        Rectangle:
            pos: self.x, self.y + self.height / 2.
            size: self.width * app.timer / 60., self.height / 2.
        Color:
            rgb: .6, .6, .6
        Rectangle:
            pos: self.x, self.y
            size: self.width * app.timer_next / 60., self.height / 2.

<GameOver>:
    BoxLayout:
        orientation: 'vertical'
        padding: '10sp'
        spacing: '10sp'

        Label:
            text: 'Game Over'
            font_size: self.height / 1.2
            size_hint_y: .10

        Label:
            text: '{}'.format(app.score)
            font_size: self.height / 1.4
            size_hint_y: .20

        Label:
            text: '1. {}\\n2. {}\\n3. {}\\n'.format(*app.highscores)
            font_size: self.height / 4.
            size_hint_y: .20

        GameOverGraph:
            id: graph
            size_hint_y: .30
            spacing: '10sp'

        JewelButton:
            text: 'Restart'
            font_size: self.height / 2
            on_release: app.start()
            size_hint_y: .20

<Sidebar@BoxLayout>:
    orientation: 'horizontal'
    canvas:
        Color:
            rgba: app.sidebar_warning_color
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: '{}'.format(app.score)
        font_size: '30dp'

    Label:
        text: '{}x'.format(app.score_multiplier)
        font_size: '30dp'

    Label:
        text: '{}'.format(app.score_combo)
        font_size: '30dp'
    #Button:
    #    text: 'Find moves'
    #    on_release: print app.board.highlight_move()


<ScreenManager>:
    canvas:
        Color:
            rgba: app.background_rgb + [.08]
        Rectangle:
            size: self.size
        PushMatrix
        Rotate:
            origin: self.width / 2., self.height / 2.
            angle: app.time * 4
            axis: 0, 0, 1
        Rectangle:
            pos: (self.width / 2.) - 256, (self.height / 2.) - 256
            size: 512, 512
            texture: app.textures['star']
        Rotate:
            origin: self.center
            angle: -app.time * 5
            axis: 0, 0, 1
        Rectangle:
            pos: (self.width / 2.) - 384, (self.height / 2.) - 384
            size: 768, 768
            texture: app.textures['star']
        Rotate:
            origin: self.center
            angle: app.time * 5
            axis: 0, 0, 1
        Rectangle:
            pos: (self.width / 2.) - 512, (self.height / 2.) - 512
            size: 1024, 1024
            texture: app.textures['star']
        Rotate:
            origin: self.center
            angle: -app.time * 6
            axis: 0, 0, 1
        Rectangle:
            pos: (self.width / 2.) - 640, (self.height / 2.) - 640
            size: 1280, 1280
            texture: app.textures['star']
        Rotate:
            origin: self.center
            angle: app.time * 6
            axis: 0, 0, 1
        Rectangle:
            pos: (self.width / 2.) - 768, (self.height / 2.) - 768
            size: 1536, 1536
            texture: app.textures['star']
        PopMatrix



<JewelUI>:
    BoxLayout:
        orientation: 'vertical'
        padding: '0dp', '5dp'
        spacing: '5dp'

        Sidebar:
            size_hint_y: None
            height: '50sp'

        Timer:
            size_hint_y: None
            height: '8sp'
            pos_hint: {'center_x': .5}

        SquareLayout:
            Board:
                id: board
''')


def print_time_spent(f):
    def f2(*args, **kwargs):
        start = time()
        ret = f(*args, **kwargs)
        print('{!r} executed in {}ms'.format(
            f.__name__, int((time() - start) * 1000)))
        return ret
    return f2


class Jewel(Widget):
    color = ListProperty([0, 0, 0])
    index = NumericProperty(0)
    ix = NumericProperty(0)
    iy = NumericProperty(0)
    board = ObjectProperty(allownone=True)
    selected = BooleanProperty(False)
    animating = BooleanProperty(False)
    anim = ObjectProperty(None, allownone=True)
    anim_highlight = ObjectProperty(None, allownone=True)

    def animate_to(self, x, y, d=0, x2=None, y2=None):
        self.animating = True
        if self.anim:
            self.anim.cancel(self)
            self.anim = None

        #distance = Vector(*self.pos).distance(Vector(x, y))
        #distance /= float(self.board.jewel_size)
        #duration = distance * .1
        duration = .20

        anim = Animation(pos=(x, y), d=duration, t='out_sine')
        if x2 is not None:
            anim = anim + Animation(pos=(x2, y2), d=duration, t='out_sine')
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

    def explode(self, *args, **kwargs):
        m = self.board.jewel_size / 2.
        anim = Animation(pos=(self.x + m, self.y + m),
                  size=(1, 1), opacity=0., d=.3)
        anim.bind(on_complete=self.destroy)
        anim.start(self)
        if 'nosound' not in kwargs:
            App.get_running_app().sound('explode')

    def explode_nosound(self, *args):
        self.explode(nosound=True)

    def destroy(self, *args):
        if self.board is not None:
            self.board.remove_widget(self)
            self.board = None

    def highlight(self):
        if self.anim_highlight:
            self.anim_highlight.cancel(self)
            self.anim_highlight = None
        d = .3
        anim_highlight = (
            Animation(opacity=.2, d=d) + Animation(opacity=1., d=d) +
            Animation(opacity=.2, d=d) + Animation(opacity=1., d=d) +
            Animation(opacity=.2, d=d) + Animation(opacity=1., d=d))
        self.anim_highlight = anim_highlight
        anim_highlight.start(self)


class TimedJewel(Jewel):
    time = NumericProperty(0)

    def __init__(self, **kwargs):
        self.time = 10 if random() < PERCENT_BIGTIME_JEWEL else 5
        super(TimedJewel, self).__init__(**kwargs)

    def explode(self, *args, **kwargs):
        self.board.app.increase_timer_next(self.time)
        super(TimedJewel, self).explode(*args)
        App.get_running_app().sound('time{}'.format(self.time))


class AreaBombJewel(Jewel):
    def explode(self, *args, **kwargs):
        super(AreaBombJewel, self).explode(*args, nosound=True)
        ix, iy = self.ix, self.iy
        jewels = []
        for x in range(max(0, ix - 1), min(SIZE, ix + 2)):
            for y in range(max(0, iy - 1), min(SIZE, iy + 2)):
                jewel = self.board.board[x][y]
                if not jewel:
                    continue
                jewels.append(jewel)
        if jewels:
            App.get_running_app().sound('explode-area')
            self.board.bam(jewels, alltogether=True, nosound=True)
            self.board.app.add_score(ix, iy, 'area', 1 + len(jewels))

class LineBombJewel(Jewel):
    def explode(self, *args, **kwargs):
        super(LineBombJewel, self).explode(*args, nosound=True)
        ix, iy = self.ix, self.iy
        jewels = []
        for x in range(0, SIZE):
            jewel = self.board.board[x][iy]
            if not jewel:
                continue
            jewels.append(jewel)

        for y in range(0, SIZE):
            jewel = self.board.board[ix][y]
            if not jewel:
                continue
            jewels.append(jewel)

        if jewels:
            App.get_running_app().sound('explode-line')
            self.board.bam(jewels, alltogether=True, nosound=True)
            self.board.app.add_score(ix, iy, 'line', 1 + len(jewels))


class SquareLayout(FloatLayout):
    def do_layout(self, *args):
        s = self.width
        if self.width > self.height:
            s = self.height
        for child in self.children:
            child.size = s, s
            child.center = self.center


class Board(StencilView):

    def __init__(self, **kwargs):
        super(Board, self).__init__(**kwargs)

        self.highlight = None
        self.app = App.get_running_app()
        self.app.board = self
        self.blocked_rows = [0] * SIZE

        # initalize the board
        self.board = []
        for index in range(SIZE):
            self.board += [[None] * SIZE]

        # fill the board
        self.first_fill = True
        self.first_run = True
        self.bind(pos=self.do_layout, size=self.do_layout)

    def reset(self, first_fill=True):
        self.clear_widgets()
        self.first_fill = first_fill
        self.blocked_rows = [0] * SIZE
        self.board = []
        for index in range(SIZE):
            self.board += [[None] * SIZE]
        Clock.schedule_once(self.fill_board, .5)

    def do_layout(self, *args):
        if self.first_fill:
            Clock.unschedule(self.fill_board)
            Clock.schedule_once(self.fill_board, .5)
            return
        js = self.jewel_size
        for ix in range(SIZE):
            for iy in range(SIZE):
                jewel = self.board[ix][iy]
                if not jewel:
                    continue
                jewel.size = js, js
                jewel.pos = self.index_to_pos(ix, iy)

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

        self.first_fill = False
        self.first_run = False


    def index_to_pos(self, ix, iy):
        js = self.jewel_size
        return self.x + ix * js, self.y + iy * js

    def touch_to_index(self, tx, ty):
        tx -= self.x
        ty -= self.y
        js = self.jewel_size
        return (
                boundary(0, SIZE, int(tx / js)),
                boundary(0, SIZE, int(ty / js)))

    @property
    def jewel_size(self):
        return int(self.width / SIZE)

    def generate(self):
        index = randint(0, 6)
        color = COLORS[index]
        js = self.jewel_size

        cls = Jewel

        if not self.first_fill and not self.app.no_touch:
            if random() < PERCENT_TIMED_JEWEL * pow(
                    DECREASE_TIMED_JEWEL, self.app.score_multiplier):
                cls = TimedJewel

        jewel = cls(index=index, color=color, board=self, size=(js, js))
        self.add_widget(jewel)
        return jewel

    def generate_at(self, ix, iy, cls=Jewel, index=None):
        index = randint(0, 6) if index is None else index
        color = COLORS[index]
        js = self.jewel_size
        jewel = cls(index=index, color=color, board=self, size=(js, js))
        jewel.ix = ix
        jewel.iy = iy
        x, y = self.index_to_pos(ix, iy)
        jewel.pos = x, y
        self.board[ix][iy] = jewel
        self.add_widget(jewel)
        return jewel

    def shuffle(self):
        dest = []
        desti = range(SIZE * SIZE)
        desti += range(SIZE * SIZE)
        #shuffle(desti)
        desti = desti[2:]
        for ix in range(SIZE):
            for iy in reversed(range(SIZE)):
                dest.append((ix, iy))

        newboard = []
        board = self.board
        for _ in range(SIZE):
            newboard += [[None] * SIZE]
        for _ix in range(SIZE):
            for _iy in range(SIZE):
                jewel = board[_ix][_iy]
                ix, iy = dest[desti.pop()]
                if not jewel:
                    jewel = self.generate()
                    jewel.ix = ix
                    jewel.iy = iy
                    newboard[ix][iy] = jewel
                    continue
                jewel.ix = ix
                jewel.iy = iy
                newboard[ix][iy] = jewel
                jewel.animate_to(*self.index_to_pos(ix, iy))
                #jewel.pos = self.index_to_pos(ix, iy)

        self.board = newboard


    def on_touch_down(self, touch):
        if self.app.no_touch:
            return
        if not self.collide_point(*touch.pos):
            return
        #if touch.is_triple_tap:
        #    self.shuffle()
            return
        ix, iy = self.touch_to_index(*touch.pos)
        jewel = self.board[ix][iy]
        if not jewel:
            return
        touch.grab(self)
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

        self.swap_with_anim(ix, iy, ix2, iy2)
        touch.ud['action'] = True

    def swap_with_anim(self, ix1, iy1, ix2, iy2):
        jewel1 = self.board[ix1][iy1]
        jewel2 = self.board[ix2][iy2]
        if jewel1 is None or jewel2 is None:
            return
        self.board[ix1][iy1] = jewel2
        self.board[ix2][iy2] = jewel1
        jewel1.ix, jewel1.iy = ix2, iy2
        jewel2.ix, jewel2.iy = ix1, iy1

        # do we have a combo anywhere ?
        h1 = self.have_combo(jewel1)
        h2 = self.have_combo(jewel2)

        # no combo ? cancel the movement
        if not (h1 or h2):
            self.board[ix1][iy1] = jewel1
            self.board[ix2][iy2] = jewel2
            jewel1.ix, jewel1.iy = ix1, iy1
            jewel2.ix, jewel2.iy = ix2, iy2

            # back and forth animation
            x1, y1 = self.index_to_pos(ix1, iy1)
            x2, y2 = self.index_to_pos(ix2, iy2)
            jewel1.animate_to(x2, y2, x2=x1, y2=y1)
            jewel2.animate_to(x1, y1, x2=x2, y2=y2)

        else:
            jewel1.animate_to(*self.index_to_pos(ix2, iy2))
            jewel2.animate_to(*self.index_to_pos(ix1, iy1))
            self.app.sound('move')

            #if PLATFORM == 'android':
            #    android.vibrate(.04)

    def highlight_move(self):
        possible_move = self.find_moves()
        if not possible_move:
            return
        ix, iy = possible_move
        jewel = self.board[ix][iy]
        jewel.highlight()

    #@print_time_spent
    def find_moves(self):
        board = self.board

        def check_moves(x2, y2):
            # FIXME reorganize to prevent too much access on board
            i22 = board[x2][y2]
            if i22 is None:
                return False
            i22_index = i22.index

            i02 = i42 = i20 = i24 = i12 = i32 = i21 = i23 = None
            if x2 >= 2:
                i02 = board[x2 - 2][y2]
            if x2 <= SIZE - 3:
                i42 = board[x2 + 2][y2]
            if y2 >= 2:
                i20 = board[x2][y2 - 2]
            if y2 <= SIZE - 3:
                i24 = board[x2][y2 + 2]
            if x2 >= 1:
                i12 = board[x2 - 1][y2]
            if x2 <= SIZE - 2:
                i32 = board[x2 + 1][y2]
            if y2 >= 1:
                i21 = board[x2][y2 - 1]
            if y2 <= SIZE - 2:
                i23 = board[x2][y2 + 1]

            return (
                (i02 is not None and i12 is not None
                    and i02.index == i12.index == i22_index) or
                (i32 is not None and i42 is not None
                    and i32.index == i42.index == i22_index) or
                (i20 is not None and i21 is not None
                    and i20.index == i21.index == i22_index) or
                (i23 is not None and i24 is not None
                    and i23.index == i24.index == i22_index) or
                (i12 is not None and i32 is not None
                    and i12.index == i32.index == i22_index) or
                (i21 is not None and i23 is not None
                    and i21.index == i23.index == i22_index))

        def swap_and_check(x1, y1, x2, y2):
            j1 = board[x1][y1]
            j2 = board[x2][y2]
            board[x1][y1] = j2
            board[x2][y2] = j1
            ret = check_moves(x1, y1)
            board[x1][y1] = j1
            board[x2][y2] = j2
            return ret

        for x in range(SIZE):
            for y in range(SIZE):
                if y >= 1:
                    if swap_and_check(x, y, x, y - 1):
                        return x, y - 1
                if y < SIZE - 1:
                    if swap_and_check(x, y, x, y + 1):
                        return x, y + 1
                if x >= 1:
                    if swap_and_check(x, y, x - 1, y):
                        return x - 1, y
                if x < SIZE - 1:
                    if swap_and_check(x, y, x + 1, y):
                        return x + 1, y


    def have_combo(self, jewel):
        sel_all, sel_x, sel_y = self.extract_combo(jewel)
        # counting
        l_all = len(sel_all)
        l_x = len(sel_x)
        l_y = len(sel_y)

        if l_all < 2:
            return
        if l_x < 2 and l_y < 2:
            return
        return True

    def extract_combo(self, jewel):
        sel_all = []
        sel_x = []
        sel_y = []
        board = self.board
        index = jewel.index
        ix, iy = jewel.ix, jewel.iy

        for x in range(ix - 1, -1, -1):
            j = board[x][iy]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_x.append(j)

        for x in range(ix + 1, SIZE):
            j = board[x][iy]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_x.append(j)

        for y in range(iy - 1, -1, -1):
            j = board[ix][y]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_y.append(j)

        for y in range(iy + 1, SIZE):
            j = board[ix][y]
            if not j or j.index != index or j.animating:
                break
            sel_all.append(j)
            sel_y.append(j)

        return sel_all, sel_x, sel_y

    def iterate_jewels(self):
        board = self.board
        for ix in range(SIZE):
            xboard = board[ix]
            for iy in range(SIZE):
                jewel = xboard[iy]
                if jewel:
                    yield ix, iy, jewel

    def check(self, jewel):
        sel_all, sel_x, sel_y = self.extract_combo(jewel)
        ix, iy = jewel.ix, jewel.iy

        # counting
        l_all = len(sel_all)
        l_x = len(sel_x)
        l_y = len(sel_y)

        if l_all < 2:
            return
        if l_x < 2 and l_y < 2:
            return

        score_pattern = 'classic'
        score_count = 1

        if l_x >= 2 and l_y >= 2:
            score_pattern = '2axes'
            jewels = [jewel] + sel_all
            index = jewel.index
            # match all colors!
            for _ix, _iy, _jewel in self.iterate_jewels():
                if _jewel.index == index and _jewel not in jewels:
                    jewels.append(_jewel)
            self.bam(jewels)
            score_count += len(jewels)

        elif l_x >= 2:
            self.bam([jewel] + sel_x)
            score_count += l_x
            if l_x == 3:
                score_pattern = '3j'
                self.generate_at(ix, iy, index=jewel.index, cls=AreaBombJewel)
            elif l_x >= 4:
                score_pattern = '4j'
                self.generate_at(ix, iy, index=jewel.index, cls=LineBombJewel)

        elif l_y >= 2:
            self.bam([jewel] + sel_y)
            score_count += l_y
            if l_y == 3:
                score_pattern = '3j'
                self.generate_at(ix, iy, index=jewel.index, cls=AreaBombJewel)
            elif l_y >= 4:
                score_pattern = '4j'
                self.generate_at(ix, iy, index=jewel.index, cls=LineBombJewel)

        self.app.add_score(ix, iy, score_pattern, score_count)


    def bam(self, jewels, alltogether=False, **kwargs):
        # first explode all the jewel
        d = 0
        board = self.board
        rows = []
        nosound = 'nosound' in kwargs
        for jewel in jewels:
            ix, iy = jewel.ix, jewel.iy
            board[ix][iy] = None

            func = jewel.explode_nosound if nosound else jewel.explode
            Clock.schedule_once(func, d)
            if not alltogether:
                d += .2
            if ix not in rows:
                rows.append(ix)
                self.blocked_rows[ix] += 1

        if alltogether:
            d = 0.2

        # more combo!
        self.app.score_combo += 1
        ix = jewels[-1].ix
        iy = jewels[-1].iy
        self.app.add_score(ix, iy, 'combo', 1)

        Clock.unschedule(self.reset_combo)
        Clock.schedule_once(self.reset_combo, d + 1.)
        Clock.schedule_once(partial(self.unblock_rows, rows), d)

    def unblock_rows(self, rows, *args):
        for row in rows:
            if self.blocked_rows[row] == 0:
                continue
            self.blocked_rows[row] -= 1
            if self.blocked_rows[row] == 0:
                self.refill(row)

    def reset_combo(self, *dt):
        self.app.reset_combo()
        if self.find_moves() is None:
            self.reset()
            self.app.show_text('No more moves')

    def levelup(self):
        for ix in range(0, SIZE):
            for iy in range(0, SIZE):
                jewel = self.board[ix][iy]
                if not isinstance(jewel, TimedJewel):
                    continue
                self.remove_widget(jewel)
                self.board[ix][iy] = None
                if jewel.time == 5:
                    cls = AreaBombJewel
                else:
                    cls = LineBombJewel
                self.generate_at(ix, iy, index=jewel.index, cls=cls)

    def gameover(self):
        for ix in range(0, SIZE):
            for iy in range(0, SIZE):
                jewel = self.board[ix][iy]
                if not isinstance(jewel, TimedJewel):
                    continue
                self.remove_widget(jewel)
                self.board[ix][iy] = None
                if jewel.time == 5:
                    cls = AreaBombJewel
                else:
                    cls = LineBombJewel
                self.generate_at(ix, iy, index=jewel.index, cls=cls)

    def refill(self, ix):

        # then make them fall down
        board = self.board

        missing = board[ix].count(None)
        if missing == 0:
            return

        row_jewels = [x for x in board[ix] if x is not None]
        board[ix] = [None] * SIZE

        iy = 0
        for jewel in row_jewels:
            if jewel.iy != iy:
                jewel.iy = iy
                jewel.animate_to(*self.index_to_pos(ix, iy))
            board[ix][iy] = jewel
            iy += 1

        for iy in range(iy, SIZE):
            jewel = self.generate()
            jewel.ix = ix
            jewel.iy = iy
            x, y = self.index_to_pos(ix, iy)
            ax, ay = self.index_to_pos(ix, iy + missing)
            jewel.pos = ax, ay
            jewel.animate_to(x, y)#, d=(iy - missing) / 10.)
            board[ix][iy] = jewel


class GameOverGraphBar(BoxLayout):
    level = NumericProperty(0)
    score = NumericProperty(0)
    alpha = NumericProperty(0)
    alphascore = NumericProperty(0)
    maxscore = NumericProperty(0)

    def animate(self):
        if self.maxscore == 0:
            return
        Animation(alphascore=self.score / float(self.maxscore),
                alpha=1.).start(self)

    def to_human(self, value):
        value = int(value)
        if value > 1000:
            value = int(value / 1000)
            return '{}K'.format(value)
        return '{}'.format(value)

class GameOverGraph(BoxLayout):

    def reset(self):
        app = App.get_running_app()
        self.clear_widgets()
        maxscore = max(app.score_levels)
        self.add_widget(Widget())
        for index in range(len(app.score_levels)):
            self.add_widget(GameOverGraphBar(level=index + 1,
                score=app.score_levels[index],
                maxscore=maxscore))
        self.add_widget(Widget())

    def animate(self):
        for bar in self.children:
            if isinstance(bar, GameOverGraphBar):
                bar.animate()

class GameOver(Screen):
    def on_pre_enter(self):
        super(GameOver, self).on_enter()
        self.ids.graph.reset()

    def on_enter(self):
        super(GameOver, self).on_enter()
        self.ids.graph.animate()


class JewelUI(Screen):
    pass


class ScoreLabel(Label):
    dy = NumericProperty(0)
    origin = ListProperty([0, 0])
    def __init__(self, **kwargs):
        super(ScoreLabel, self).__init__(**kwargs)
        anim = Animation(dy=sp(50), opacity=0., d=.5)
        anim.bind(on_complete=self.remove_parent)
        anim.start(self)

    def remove_parent(self, *args):
        if self.parent:
            self.parent.remove_widget(self)


class TitleText(Label):

    def __init__(self, **kwargs):
        super(TitleText, self).__init__(**kwargs)
        self.opacity = 0.
        self.pos_hint = {'center_y': 0.}
        anim = (
            Animation(opacity=1., d=.5, pos_hint={'center_y': .50}, t='out_quad') +
            Animation(opacity=0., d=.5, pos_hint={'center_y': 1.}, t='in_quad'))
        anim.start(self)


class JewelApp(App):
    score = NumericProperty(0)

    score_multiplier = NumericProperty(1)

    score_combo = NumericProperty(0)

    timer = NumericProperty(LEVEL_TIME)

    level_time = NumericProperty(LEVEL_TIME)

    timer_next = NumericProperty(0)

    no_touch = BooleanProperty(False)

    highscores = ListProperty([0, 0, 0])

    background_rgb = ListProperty([0, 0, 0])

    sidebar_warning_color = ListProperty([0, 0, 0, 0])

    time = NumericProperty(0)

    score_levels = ListProperty([0])

    gameover_graph = ObjectProperty()

    timer_hint = NumericProperty(0)

    def build(self):
        self.textures = {}
        self.load_sounds()
        self.highscore_fn = join(self.user_data_dir, 'highscore.dat')

        from kivy.base import EventLoop
        EventLoop.ensure_window()
        # load textures
        for fn in ('gem', 'gem_selected', 't5', 't10', 'tarea', 'tline', 'star'):
            texture = CoreImage(join('data', '{}.png'.format(fn)), mipmap=True).texture
            self.textures[fn] = texture


        self.root = ScreenManager(transition=SlideTransition())

        self.bind(score_combo=self.check_game_over,
                timer=self.check_game_over,
                timer_next=self.check_game_over)
        self.ui_jewel = JewelUI(name='jewel')
        self.root.add_widget(self.ui_jewel)
        self.start()

        Clock.schedule_interval(self.update_time, 1 / 20.)
        #Clock.schedule_interval(self._stats, 1 / 60.)
        #self._stats()

        if False:
            def debug_leaks(*args):
                import gc; gc.collect()
                print 'collect()', len(gc.get_objects())
                import objgraph;
                #print len(objgraph.by_type('PropertyStorage'))
                objgraph.show_growth()
                #print objgraph.typestats(objgraph.get_leaking_objects())
                #objgraph.show_most_common_types()
            Clock.schedule_interval(debug_leaks, 1.)

        # load highscores
        if not exists(self.highscore_fn):
            return
        try:
            with open(self.highscore_fn) as fd:
                version, highscores = json.load(fd)
            if version == HIGHSCORE_VERSION:
                self.highscores = highscores
        except:
            pass

    def save_highscore(self):
        highscores = self.highscores + [self.score]
        highscores.sort()
        highscores = list(reversed(highscores))[:3]
        self.highscores = highscores
        with open(self.highscore_fn, 'w') as fd:
            json.dump([HIGHSCORE_VERSION, self.highscores], fd)

    def start(self):
        if not self.board.first_run:
            self.board.reset()

        self.score = 0
        self.score_combo = 0
        self.score_multiplier = 1
        self.timer = LEVEL_TIME
        self.level_time = LEVEL_TIME
        self.start_time = self.timer_hint = time()
        self.no_touch = False
        self.score_levels = [0]
        self.audio_warning_index = -1

        self.root.current = 'jewel'

        Clock.schedule_interval(self.update_timer, 1 / 20.)

        self.bind(score_multiplier=self.update_background)
        self.update_background()

        self.show_text('Level 1')

    def update_background(self, *args):
        index = (self.score_multiplier - 1) % len(BACKGROUND_COLORS)
        c = BACKGROUND_COLORS[index]
        Animation(background_rgb=c).start(self)

    def game_over(self):
        self.no_touch = True
        self.timer = 0
        self.board.gameover()
        self.sound('end')
        Clock.unschedule(self.update_timer)

    def check_game_over(self, *args):
        if any([self.score_combo, self.timer, self.timer_next]):
            return
        if self.no_touch:
            self.save_highscore()
            if not self.root.has_screen('gameover'):
                self.gameover = GameOver(name='gameover')
                self.root.add_widget(GameOver(name='gameover'))
            self.root.current = 'gameover'

    def update_time(self, dt):
        self.time += dt

    def update_timer(self, dt):
        self.sound_played = []
        current_time = time()
        self.timer = self.level_time - (current_time - self.start_time)

        should_hint = current_time - self.timer_hint
        if should_hint > HINT_TIME:
            self.timer_hint = current_time
            if self.score_combo == 0:
                self.board.highlight_move()

        t = self.timer + self.timer_next
        if t < 10:
            a = abs(sin(self.time * 3))
            self.sidebar_warning_color = [1, .1, .1,
                    ((10. - t) / 10.) * .3 + a * .5]

            audio_warning_index = 10 - int(t)
            if audio_warning_index != self.audio_warning_index:
                self.play_sound('warning',
                        volume=(audio_warning_index / 15.))
                self.audio_warning_index = audio_warning_index

        else:
            self.sidebar_warning_color = [0, 0, 0, 0]
            self.audio_warning_index = -1


        if self.timer > 0:
            return

        if self.timer_next == 0:
            self.game_over()
            return

        # next level!
        self.score_multiplier += 1
        self.start_time = time()
        self.level_time = self.timer_next
        self.timer = self.level_time - (time() - self.start_time)
        self.timer_next = self.property('timer_next').defaultvalue
        self.score_levels.append(0)
        self.board.levelup()
        self.show_text('Level {}'.format(self.score_multiplier))
        self.sound('levelup')


    def add_score(self, ix, iy, pattern, count):
        x, y = self.board.index_to_pos(ix, iy)

        m = self.score_multiplier
        score = 0
        if pattern == 'classic':
            score += count * 50
        elif pattern == '4j':
            score += count * 100
        elif pattern == '5j':
            score += count * 150
        elif pattern == '2axes':
            score += count * 200
        elif pattern == 'area':
            score += count * 500
        elif pattern == 'line':
            score += count * 1000
        elif pattern == 'combo':
            score += self.score_combo * 1000

        score *= m
        self.score += score
        self.score_levels[-1] += score

        if score == 0:
            return

        js = self.board.jewel_size
        label = ScoreLabel(text='{}'.format(score), origin=(x + js / 2, y + js / 2))
        self.ui_jewel.add_widget(label)

    def increase_timer_next(self, t):
        if self.no_touch:
            return
        self.timer_next = min(LEVEL_TIME, t + self.timer_next)

    def reset_combo(self):
        self.score_combo = 0
        self.timer_hint = time()

    def show_text(self, text):
        ttext = TitleText(text=text)
        self.ui_jewel.add_widget(ttext)

    def load_sounds(self):
        self.music = SoundLoader.load('data/audio/Tetra_-_Hunters.ogg')
        self.music.volume = .35
        self.music.loop = True
        self.music.play()

        self.sounds = {}
        self.load_sound('move', 'move.ogg', count=5)
        self.load_sound('levelup', 'levelup.ogg')
        self.load_sound('end', 'end.ogg')
        self.load_sound('time5', 'time5.ogg', count=2)
        self.load_sound('time10', 'time10.ogg', count=2)
        self.load_sound('warning', 'warning.ogg')
        self.load_sound('explode-line', 'explode-line.ogg', count=2)
        self.load_sound('explode-area', 'explode-area.ogg', count=2)

    def load_sound(self, action, fn, count=1):
        sounds = [SoundLoader.load('data/audio/{}'.format(fn)) for x in range(count)]
        self.sounds[action] = sounds

    def play_sound(self, action, volume=1.):
        if action in self.sound_played:
            return
        self.sound_played.append(action)
        sounds = self.sounds[action]
        sound = sounds.pop(0)
        sound.volume = volume
        sound.play()
        sounds.append(sound)

    def sound(self, action):
        # no sound for explode yet.
        if action == 'explode':
            action = 'move'
        self.play_sound(action)

    '''
    def _stats(self, *dt):
        if not hasattr(self, '_stat_frame'):
            self._stat_frame = 0
            self._stat_lastdiff = 0
            self._stat_time = time()
            self._stat_profile = None
            return

        if self._stat_frame % 100 == 0:
            if self._stat_profile is not None:
                self._stat_profile.disable()
                self._stat_profile.create_stats()
                fn = '/sdcard/frame{}'.format(self._stat_frame)
                self._stat_profile.dump_stats(fn)
                print '-- dump stats to', fn

            self._stat_profile = profile.Profile()
            self._stat_profile.enable()

        current = time()
        diff = (current - self._stat_time) * 1000
        warning = None
        if abs(diff - self._stat_lastdiff) > self._stat_lastdiff * 0.5:
            warning = abs(diff - self._stat_lastdiff)
        self._stat_lastdiff = diff
        print 'Frame {}: {:.4f} {}'.format(self._stat_frame, diff,
                'WARNING {}ms'.format(warning) if warning is not None else '')
        self._stat_frame += 1
        self._stat_time = current
    '''


JewelApp().run()
