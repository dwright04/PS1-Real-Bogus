import wx, sys, optparse, time
import scipy.io as sio
import numpy as np
# The recommended way to use wx with mpl is with the WXAgg
# backend. 
#
import wx.lib.agw.hyperlink as hl
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar
    
from TargetImage import TargetImage
from classify import predict
from profiling import profileData

sys.path.insert(1,"/Users/dew/development/PS1-Real-Bogus/data/")
from combine_data_sets import loadData
from build_data_set import process_examples, build_data_set, signPreserveNorm, bg_sub_signPreserveNorm, noNorm

sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/demos/")
import mlutils
PATH = "/Users/dew/myscripts/machine_learning/data/3pi/detectionlist/"

"""
import os
import cPickle as pickle
from functools import wraps

class Cached(object):
    def __init__(self, filename):
        self.filename = filename
    
    def __call__(self, func):
        @wraps(func)
        def new_func(*args, **kwargs):
            if not os.path.exists(self.filename):
                results = func(*args, **kwargs)
                with open(self.filename, 'w') as outfile:
                    pickle.dump(results, outfile, pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.filename, 'r') as infile:
                    results = pickle.load(infile)
            return results
        return new_func
"""
np.seterr(all="ignore")

class NavigationControlBox(wx.Panel):
    def __init__(self, parent, frame, ID, label):
        wx.Panel.__init__(self, parent, ID)
        
        self.frame = frame
        #self.frame = parent
        box = wx.StaticBox(self, -1, label)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.next_button = wx.Button(self, -1, label="Next 100")
        self.next_button.Bind(wx.EVT_BUTTON, self.on_next)
        self.previous_button = wx.Button(self, -1, label="Previous 100")
        self.previous_button.Bind(wx.EVT_BUTTON, self.on_previous)
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.previous_button, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.AddSpacer(10)
        manual_box.Add(self.next_button, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def on_next(self, event):
        #for ax in self.frame.AXES:
        #    try:
        #        for i in range(4):
        #            ax.lines.pop(0)
        #    except IndexError:
        #        pass
        self.frame.start += 100
        self.frame.end += 100
        if self.frame.start > 0:
            self.frame.navigation_control.previous_button.Enable()
        if self.frame.end == self.frame.max_index:
            self.frame.navigation_control.next_button.Disable()
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        print "next: %d-%d" % (self.frame.start+1,self.frame.end)
            
    def on_previous(self, event):
        #for ax in self.frame.AXES:
        #    try:
        #        for i in range(4):
        #            ax.lines.pop(0)
        #    except IndexError:
        #        pass
        if self.frame.start <=0:
            self.frame.navigation_control.previous_button.Disable()
        else:
            self.frame.start -= 100
            self.frame.end -= 100
            print self.frame.start, self.frame.end
            if self.frame.start ==0:
                self.frame.navigation_control.previous_button.Disable()
            if self.frame.start < self.frame.max_index:
                self.frame.navigation_control.next_button.Enable()
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        print "previous: %d-%d" % (self.frame.start+1,self.frame.end)

class LabelKeyBox(wx.Panel):
    def __init__(self, parent, ID):
        wx.Panel.__init__(self, parent, ID)
        

        box = wx.StaticBox(self, -1, "relabelling key")
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        self.real = wx.StaticText(self, label="REAL")
        self.real.SetBackgroundColour(wx.WHITE)
        self.real.SetForegroundColour("#3366FF")

        self.bogus = wx.StaticText(self, label="BOGUS")
        self.bogus.SetBackgroundColour(wx.WHITE)
        self.bogus.SetForegroundColour("#FF0066")

        self.ghost = wx.StaticText(self, label="GHOST")
        self.ghost.SetBackgroundColour(wx.WHITE)
        self.ghost.SetForegroundColour("#9933FF")

        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.real, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.bogus, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.ghost, flag=wx.ALIGN_CENTER_VERTICAL)

        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)

class DataSetControlBox(wx.Panel):
    def __init__(self, parent, frame, ID):
        wx.Panel.__init__(self, parent, ID)
        
        self.frame = frame
        box = wx.StaticBox(self, -1, "data set control")
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.fp_button = wx.Button(self, -1, label="FP")
        self.fp_button.Bind(wx.EVT_BUTTON, self.on_fp)
        self.tp_button = wx.Button(self, -1, label="TP")
        self.tp_button.Bind(wx.EVT_BUTTON, self.on_tp)
        self.fn_button = wx.Button(self, -1, label="FN")
        self.fn_button.Bind(wx.EVT_BUTTON, self.on_fn)
        self.tn_button = wx.Button(self, -1, label="TN")
        self.tn_button.Bind(wx.EVT_BUTTON, self.on_tn)
        
        self.real_button = wx.Button(self, -1, label="Real")
        self.real_button.Bind(wx.EVT_BUTTON, self.on_real)
        self.bogus_button = wx.Button(self, -1, label="Bogus")
        self.bogus_button.Bind(wx.EVT_BUTTON, self.on_bogus)
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.tp_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.fp_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.tn_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.fn_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        
        manual_box.Add(self.real_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.bogus_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)

    def on_tp(self, event):
        print "true positives"
        self.frame.X = self.frame.dataSetDict["true_pos_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["true_pos_y"]
        self.frame.files = self.frame.dataSetDict["true_pos_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.previous_button.Disable()
        self.frame.navigation_control.next_button.Enable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : True Positives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Disable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_fp(self, event):
        print "false positives"
        self.frame.X = self.frame.dataSetDict["false_pos_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["false_pos_y"]
        self.frame.files = self.frame.dataSetDict["false_pos_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.previous_button.Disable()
        self.frame.navigation_control.next_button.Enable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : False Positives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Disable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_tn(self, event):
        print "true negatives"
        self.frame.X = self.frame.dataSetDict["true_neg_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["true_neg_y"]
        self.frame.files = self.frame.dataSetDict["true_neg_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : True Negatives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Disable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_fn(self, event):
        print "false negatives"
        self.frame.X = self.frame.dataSetDict["false_neg_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["false_neg_y"]
        self.frame.files = self.frame.dataSetDict["false_neg_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : Fasle Negative (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Disable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()

    def on_real(self, event):
        print "show real"
        self.frame.X = self.frame.dataSetDict["real_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["real_y"]
        self.frame.files = self.frame.dataSetDict["real_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : Real (%d examples)" % m)
        self.frame.data_set_control.real_button.Disable()
        self.frame.data_set_control.bogus_button.Enable()

    def on_bogus(self, event):
        print "show bogus"
        self.frame.X = self.frame.dataSetDict["bogus_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/100.0)*100)
        self.frame.y = self.frame.dataSetDict["bogus_y"]
        self.frame.files = self.frame.dataSetDict["bogus_files"]
        self.frame.start = 0
        self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : Bogus (%d examp les)" % m)
        self.frame.data_set_control.bogus_button.Disable()
        self.frame.data_set_control.real_button.Enable()

class mainFrame(wx.Frame):
    """ The main frame of the application
    """
    title = 'Main Console'
    def __init__(self, dataSetDict, infoDict):
        # 10x10 inches, 100 dots-per-inch, so set size to (1000,1000)
        wx.Frame.__init__(self, None, -1, self.title, size=(1000,1000),pos=(50,50))
        
        self.dataSetDict = dataSetDict
        self.infoDict = infoDict
        self.X = self.dataSetDict["X"]
        self.y = self.dataSetDict["y"]
        self.files = self.dataSetDict["files"]
        m, n = np.shape(self.X)
        self.plotDim = np.sqrt(n)
        self.info = infoDict
        self.start = 0
        self.end = 100
        self.to_plot = self.X[self.start:self.end,:]
        self.files_to_plot = self.files[self.start:self.end]
        print len(self.files_to_plot)
        self.max_index = int(np.ceil(np.shape(self.X)[0]/100.0)*100)

        self.new_real_files = []
        self.new_bogus_files = []
        self.new_ghost_files = []
        
        # Create the mpl Figure and FigCanvas objects.
        # 10x10 inches, 100 dots-per-inch
        #

        self.dpi = 100
        self.fig = Figure((8.0, 8.0), dpi=self.dpi)
        self.fig.subplots_adjust(left=0.1, bottom=0.01, right=0.9, top=0.99, wspace=0.05, hspace=0.05)

        self.AXES = []
        for i in range(100):
            ax = self.fig.add_subplot(10,10,i+1)
            self.AXES.append(ax)
        
        self.create_main_panel()
        #self.create_main_panel()
        self.navigation_control.previous_button.Disable()
        
        if "true_pos_X" not in self.dataSetDict.keys():
            self.data_set_control.tp_button.Disable()
            self.data_set_control.fp_button.Disable()
            self.data_set_control.tn_button.Disable()
            self.data_set_control.fn_button.Disable()
        
    def create_main_panel(self):

        self.panel = wx.Panel(self)
        m, n = np.shape(self.X)
        self.set_text = wx.StaticText(self.panel, -1, label="Showing : All (%d examples)" % m)
        self.set_text.SetBackgroundColour(wx.WHITE)
        font = wx.Font(20, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.set_text.SetFont(font)
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.set_text, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

        self.draw_fig(True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        # Bind the 'click' event for clicking on one of the axes
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.navigation_control = NavigationControlBox(self.panel, self, -1, "navigation control")
        self.label_key_box = LabelKeyBox(self.panel,-1)
        self.data_set_control = DataSetControlBox(self.panel,self,-1)
        
        
        self.build_button = wx.Button(self.panel, -1, label="Build")
        self.build_button.Bind(wx.EVT_BUTTON, self.on_build)
        self.reset_button = wx.Button(self.panel, -1, label="Reset")
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset)
        self.exit_button = wx.Button(self.panel, -1, label="Exit")
        self.exit_button.Bind(wx.EVT_BUTTON, self.on_exit)

        self.vbox1 = wx.BoxSizer(wx.VERTICAL)
        self.vbox1.Add(self.build_button, 0, flag=wx.CENTER | wx.BOTTOM)
        self.vbox1.Add(self.reset_button, 0, flag=wx.CENTER | wx.BOTTOM)
        self.vbox1.Add(self.exit_button, 0, flag=wx.CENTER | wx.BOTTOM)
        #self.panel.SetSizer(self.vbox1)
        #self.vbox1.Fit(self)
        
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2.Add(self.label_key_box, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.data_set_control, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.navigation_control, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.vbox1, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        #text = wx.StaticText(self.panel, -1, label="TEST")
        #self.hbox2.Add(text, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        
        self.vbox2 = wx.BoxSizer(wx.VERTICAL)
        self.vbox2.Add(self.hbox1, 0, flag=wx.CENTER | wx.TOP)
        self.vbox2.Add(self.canvas, 1, flag=wx.CENTER | wx.CENTER | wx.GROW)
        self.vbox2.Add(self.hbox2, 0, flag=wx.LEFT | wx.TOP)
        
        self.panel.SetSizer(self.vbox2)
        self.vbox2.Fit(self)

    def draw_fig(self, init=False):

        #matplotlib.pyplot.clf()
        #print self.files_to_plot[0].rstrip().split("/")[-1]
        #print set(self.new_bogus_files)
        for i,ax in enumerate(self.AXES):
            cmap="hot"
            if init:
                try:
                    image = np.reshape(self.to_plot[i,:], (self.plotDim, self.plotDim), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
                except IndexError:
                    ax.clear()
                    image = np.reshape(np.zeros((400,)), (self.plotDim, self.plotDim), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
            try:
                #print self.files_to_plot[i].rstrip().split("/")[-1] in set(self.new_bogus_files)
                if self.files_to_plot[i].rstrip().split("/")[-1] in set(self.new_real_files):
                    ax.clear()
                    cmap="cool"
                    image = np.reshape(self.to_plot[i,:], (self.plotDim, self.plotDim), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
                elif self.files_to_plot[i].rstrip().split("/")[-1] in set(self.new_bogus_files):
                    ax.clear()
                    cmap = "PRGn"
                    image = np.reshape(self.to_plot[i,:], (self.plotDim, self.plotDim), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
            except IndexError:
                ax.clear()
                image = np.reshape(np.zeros((400,)), (self.plotDim, self.plotDim), order="F")
                image = np.flipud(image)
                ax.imshow(image, interpolation="nearest", cmap=cmap)
                ax.axis("off")
            #if color != None:
            #    ax.plot([0, 20], [0, 0], color=color, lw=3)
            #    ax.plot([0, 0], [0, 20.5], color=color, lw=3)
            #    ax.plot([0, 20], [20.5, 20.5], color=color, lw=3)
            #    ax.plot([20.5, 20.5], [0, 20.5], color=color, lw=3)

    def on_click(self, event):
        """Enlarge or restore the selected axis."""
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.
        # 
        if hasattr(self, "inspectorFrame"):
            return
        
        self.axes = event.inaxes
        if self.axes is None:
            return

        #try:
        #    self.axes.lines[0]
        #except IndexError:
        #    self.axes.plot([0, 20], [0, 0], color="#66FF33", lw=3)
        #    self.axes.plot([0, 0], [0, 20.5], color="#66FF33", lw=3)
        #    self.axes.plot([0, 20], [20.5, 20.5], color="#66FF33", lw=3)
        #    self.axes.plot([20.5, 20.5], [0, 20.5], color="#66FF33", lw=3)

        self.canvas.draw()
        self.canvas.Refresh()

        file = self.files_to_plot[self.AXES.index(self.axes)]
        
        if event.button is 1:

            if file.split("/")[-1] in set(self.new_real_files):
                label = 1
            elif file.split("/")[-1] in set(self.new_bogus_files):
                label = 0
            elif file.split("/")[-1] in set(self.new_ghost_files):
                label = 2
            else:
                label = self.y[self.start+self.AXES.index(self.axes)]
            print self.info.keys()[0]
            try:
                info = self.info[file]
                info.append(self.files_to_plot.index(file))
                info.append(label)
            except KeyError:
                info = []
            if len(info) > 2:
                infoFlag = True
            print info
            self.inspectorFrame = InspectorFrame(self, info, infoFlag)
            self.inspectorFrame.Show()
        elif event.button is 3:
            label = self.y[self.start+self.AXES.index(self.axes)]
            if label == 0:
                self.new_real_files.append(file.split("/")[-1])
            elif label == 1:
                self.new_bogus_files.append(file.split("/")[-1])
            self.draw_fig()
            self.canvas.draw()
            self.canvas.Refresh()

    def on_build(self, event):
        X = self.dataSetDict["X"]
        y = np.squeeze(self.dataSetDict["y"])
        print np.shape(y)
        for i,file in enumerate(self.dataSetDict["files"]):
            print file.rstrip().split("/")[-1], y[i],
            if file.rstrip().split("/")[-1] in self.new_bogus_files:
                if y[i] == 1:
                    y[i] = 0
            if file.rstrip().split("/")[-1] in self.new_real_files:
                if y[i] == 0:
                    y[i] = 1
            print y[i]
        outputFile = raw_input("Specify output file : ")
        sio.savemat("reviewed_data_sets/"+outputFile, {"X": X, "y":y, "files": self.dataSetDict["files"]})
        print "[+] Processing complete."
        
        """
        norm = raw_input("Specify normalisation function to apply to data [default=signPreserveNorm] : ")
        
        if norm == "":
            norm = "spn"
        
        if norm == "signPreserveNorm" or norm == "spn":
            normFunc = signPreserveNorm
        elif norm == "bg_sub_signPreserveNorm" or norm == "bg_sub_spn":
            normFunc = bg_sub_signPreserveNorm
        elif norm == "noNorm":
            normFunc = noNorm

        extent = raw_input("Specify image size [default=10] : ")

        if extent == "":
            extent = 10

        path = raw_input("Specify path to data set detectionList directory : ")

        startTime = time.time()
        
        y = self.dataSetDict["y"]

        pos_list = self.dataSetDict["real_files"]
        neg_list = self.dataSetDict["bogus_files"]

        print "[+] Processing positve examples."
        for file in pos_list[:]:
            if file in set(self.new_bogus_files):
                pos_list.remove(file)
            elif file in set(self.new_ghost_files):
                pos_list.remove(file)
            elif file in set(self.new_real_files) and file not in set(pos_list):
                pos_list.append(file)
        m_pos = len(pos_list)
        pos_data = process_examples(pos_list, path, 1, int(extent), normFunc)
        posX = np.concatenate((pos_data[0], pos_data[3]))
        posy = np.concatenate((pos_data[1], pos_data[4]))
        posfiles = np.concatenate((pos_data[2], pos_data[5]))
        pos_data = (posX, posy, posfiles)
        print "[+] %d positive examples processed." % m_pos
            
        print "[+] Processing negative examples."
        for file in neg_list[:]:
            if file in set(self.new_real_files):
                neg_list.remove(file)
            elif file in set(self.new_ghost_files):
                neg_list.remove(file)
            elif file in set(self.new_bogus_files) and file not in set(neg_list):
                neg_list.append(file)
        m_neg = len(neg_list)
        neg_data = process_examples(neg_list, path, 1, int(extent), normFunc)
        negX = np.concatenate((neg_data[0], neg_data[3]))
        negy = np.concatenate((neg_data[1], neg_data[4]))
        negfiles = np.concatenate((neg_data[2], neg_data[5]))
        neg_data = (negX, negy, negfiles)
        print "[+] %d negative examples processed." % m_neg

        print "[+] Building training set."
        X, y, train_files = build_data_set(pos_data, neg_data)

        outputFile = raw_input("Specify output file : ")
        print "[+] Saving data set in ."
        sio.savemat("reviewed_data_sets/"+outputFile, {"X": X, "y":y, "files": train_files})
        print "[+] Processing complete."
        print "[*] Run time: %d minutes." % ((time.time() - startTime) / 60)
        """
    def on_reset(self, event):
        
        self.new_real_files = []
        self.new_bogus_files = []
        self.new_ghost_files = []
    
    def on_exit(self, event):
        self.Destroy()
        exit(0)
    
class LabellingControlBox(wx.Panel):
    def __init__(self, parent, frame, ID, label, title):
        wx.Panel.__init__(self, parent, ID)
        self.parent = parent
        self.frame = frame
        
        box = wx.StaticBox(self, -1, title)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.real_button = wx.Button(self, -1, label="real")
        self.real_button.Bind(wx.EVT_BUTTON, self.on_real)
        self.bogus_button = wx.Button(self, -1, label="bogus")
        self.bogus_button.Bind(wx.EVT_BUTTON, self.on_bogus)
        self.ghost_button = wx.Button(self, -1, label="ghost")
        self.ghost_button.Bind(wx.EVT_BUTTON, self.on_ghost)
        self.cancel_button = wx.Button(self, -1, label="cancel")
        self.cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        
        if int(label)  == 1:
            self.real_button.Disable()
        elif int(label) == 0:
            self.bogus_button.Disable()
        elif int(label) == 2:
            self.ghost_button.Disable()
        
        
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.real_button, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.AddSpacer(10)
        manual_box.Add(self.bogus_button, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.AddSpacer(10)
        manual_box.Add(self.ghost_button, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.AddSpacer(10)
        manual_box.Add(self.cancel_button, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def on_real(self, event):
        print "real"
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_bogus_files):
            self.frame.caller.new_bogus_files.remove(self.frame.file.split("/")[-1])
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_ghost_files):
            self.frame.caller.new_ghost_files.remove(self.frame.file.split("/")[-1])
        self.frame.caller.new_real_files.append(self.frame.file.split("/")[-1])
        #for i in range(4):
        #    self.frame.caller.axes.lines.pop(0)
        #self.frame.caller.axes.plot([0, 20], [0, 0], color="#3366FF", lw=3)
        #self.frame.caller.axes.plot([0, 0], [0, 20.5], color="#3366FF", lw=3)
        #self.frame.caller.axes.plot([0, 20], [20.5, 20.5], color="#3366FF", lw=3)
        #self.frame.caller.axes.plot([20.5, 20.5], [0, 20.5], color="#3366FF", lw=3)
        self.frame.caller.draw_fig()
        self.frame.caller.canvas.draw()
        self.frame.caller.canvas.Refresh()
        self.frame.Destroy()
        del self.frame.caller.inspectorFrame
    
    def on_bogus(self, event):
        print "bogus"
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_real_files):
            self.frame.caller.new_real_files.remove(self.frame.file.split("/")[-1])
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_ghost_files):
            self.frame.caller.new_ghost_files.remove(self.frame.file.split("/")[-1])
        self.frame.caller.new_bogus_files.append(self.frame.file.split("/")[-1])
        #for i in range(4):
        #    self.frame.caller.axes.lines.pop(0)
        #self.frame.caller.axes.plot([0, 20], [0, 0], color="#FF0066", lw=3)
        #self.frame.caller.axes.plot([0, 0], [0, 20.5], color="#FF0066", lw=3)
        #self.frame.caller.axes.plot([0, 20], [20.5, 20.5], color="#FF0066", lw=3)
        #self.frame.caller.axes.plot([20.5, 20.5], [0, 20.5], color="#FF0066", lw=3)
        self.frame.caller.draw_fig()
        self.frame.caller.canvas.draw()
        self.frame.caller.canvas.Refresh()
        self.frame.Destroy()
        del self.frame.caller.inspectorFrame
        
    def on_ghost(self, event):
        print "ghost"
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_bogus_files):
            self.frame.caller.new_bogus_files.remove(self.frame.file.split("/")[-1])
        if self.frame.file.split("/")[-1] in set(self.frame.caller.new_real_files):
            self.frame.caller.new_real_files.remove(self.frame.file.split("/")[-1])
        self.frame.caller.new_ghost_files.append(self.frame.file.split("/")[-1])
        #for i in range(4):
        #    self.frame.caller.axes.lines.pop(0)
        #self.frame.caller.axes.plot([0, 20], [0, 0], color="#9933FF", lw=3)
        #self.frame.caller.axes.plot([0, 0], [0, 20.5], color="#9933FF", lw=3)
        #self.frame.caller.axes.plot([0, 20], [20.5, 20.5], color="#9933FF", lw=3)
        #self.frame.caller.axes.plot([20.5, 20.5], [0, 20.5], color="#9933FF", lw=3)
        self.frame.caller.draw_fig()
        self.frame.caller.canvas.draw()
        self.frame.caller.canvas.Refresh()
        self.frame.Destroy()
        del self.frame.caller.inspectorFrame
        
    def on_cancel(self, event):
        print "cancel"

        #print self.frame.caller.axes.lines[0].get_color()
        #if self.frame.caller.axes.lines[0].get_color() == "#66FF33":
        #    for i in range(4):
        #        self.frame.caller.axes.lines.pop(0)
        #self.frame.caller.canvas.draw()
        #self.frame.caller.draw_fig()
        #self.frame.caller.canvas.Refresh()
        self.frame.Destroy()
        del self.frame.caller.inspectorFrame
        
class InfoBox(wx.Panel):
    def __init__(self, parent, info, ID, label):
        wx.Panel.__init__(self, parent, ID)
        try:
            self.id = info[0]
            self.mjd = info[1]
            if info[2] == "(ariablestar)":
                self.type = "(variablestar)"
            else:
                self.type = info[2]
            self.filter = info[3]
            self.mag = info[4]
            self.list = info[5]
            self.file = info[7]
            self.label = info[8]
            print len(info)
            self.text = "id: %s\t label: %d\nMJD: %.3f\t filter: %s\n mag: %s\t type: %s\n list: %s" \
                        % (self.id, int(self.label), float(self.mjd), self.filter, \
                           self.mag, self.type, self.list)
        except IndexError:
            self.text = "No Information Available"
        self.content = wx.StaticText(self, -1, self.text, style=wx.ALIGN_LEFT)

        box = wx.StaticBox(self, -1, label)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.content, flag=wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)
        
class InspectorFrame(wx.Frame):
    """ The inspector frame of the application
    """
    title = 'inspector frame'
    def __init__(self, caller, info, infoFlag=False):
        # 8x8 inches, 100 dots-per-inch, so set size to (800,800)
        wx.Frame.__init__(self, None, -1, self.title, size=(800,800))
        self.caller = caller
        self.info = info
        #print self.info
        self.file = PATH + str(int(info[-1])) + "/" + info[-3].split("/")[-1]
        #print self.caller.to_plot
        """
        if infoFlag:
            self.file = info[6].split("/")[-1].strip()
            self.smallImage = self.caller.to_plot[self.caller.files_to_plot.index(self.file),:]
        else:
            self.file = info[0]
            self.smallImage = self.caller.to_plot[self.info[1],:]
        """
        #print self.file
        try:
            self.smallImage = np.nan_to_num(TargetImage(self.file, 10).signPreserveNorm())
        except IOError:
            try:
                self.file = PATH + "2" + "/" + info[-3].split("/")[-1]
                self.smallImage = np.nan_to_num(TargetImage(self.file, 10).signPreserveNorm())
            except IOError, e:
                print e
        try:
            self.largeImage = np.nan_to_num(TargetImage(self.file, 50).unravelObject())
        except IOError:
            self.largeImage = self.smallImage
        self.create_main_panel()
    
    def create_main_panel(self):
        self.panel = wx.Panel(self)
        
        # Create the mpl Figure and FigCanvas objects. 
        # 8x8 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = Figure((7.0, 7.0), dpi=self.dpi)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.visualiseImage()
        
        #id = self.file.split("/")[-1].split("_")[0]
        
        #self.hyperlink = hl.HyperLinkCtrl(self.panel, -1, "object web page", pos=(100, 100),
        #                                  URL="http://star.pst.qub.ac.uk/sne/ps1md/psdb/candidate/"+id+"/")
        
        self.labelling_control = LabellingControlBox(self.panel, self,-1, self.info[-1], "Labelling Control")
        
        self.info_box = InfoBox(self.panel, self.info, -1, "Additional Information")
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.labelling_control, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.info_box, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        #self.hbox1.Add(self.hyperlink, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        
        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)        
        
    def visualiseImage(self):
        extentLarge = int(0.5*np.sqrt(len(self.largeImage)))
        extent = int(0.5*np.sqrt(len(self.smallImage)))

        ax1 = self.fig.add_subplot(2,2,1)
        ax1.set_xlabel("pixel")
        ax1.set_xticks([0.5,extent-0.5,2*extent-0.5])
        ax1.set_xticklabels([1,extent,2*extent])
        ax1.set_ylabel("pixel")
        ax1.set_yticks([0.5,extent-0.5,2*extent-0.5])
        ax1.set_yticklabels([1,extent,2*extent])
        
        ImageSmall = np.reshape(self.smallImage, (int(2*extent),int(2*extent)), order="F")
        im1 = ax1.imshow(ImageSmall, interpolation="nearest", cmap="hot", origin="lower")
        ax1.plot(extent-0.5, extent-0.5, "k+")
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax1)
        # Append axes to the right of ax3, with 20% width of ax1
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        #
        cbar = plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.5))

        ax2 = self.fig.add_subplot(2,2,3)        
        ax2.set_xlabel("pixel")
        ax2.set_xticks([0.5,extentLarge-0.5,2*extentLarge-0.5])
        ax2.set_xticklabels([1,extentLarge,2*extentLarge])
        ax2.set_ylabel("pixel")
        ax2.set_yticks([0.5,extentLarge-0.5,2*extentLarge-0.5])
        ax2.set_yticklabels([1,extentLarge,2*extentLarge])
    
        try:
            ImageLarge = np.reshape(self.largeImage, (int(2*extentLarge),int(2*extentLarge)), order="F")
        except:
            ImageLarge = np.reshape(self.largeImage, (100,100), order="F")
        im2 = ax2.imshow(ImageLarge, interpolation="nearest", cmap="hot", origin="lower")
        ax2.plot(np.arange(0,2*extentLarge), (extentLarge-0.5)*np.ones((2*extentLarge,)), "k-")
        ax2.plot((extentLarge-0.5)*np.ones((2*extentLarge,)), np.arange(0,2*extentLarge), "k-")
        # Create divider for existing axes instance
        divider2 = make_axes_locatable(ax2)
        # Append axes to the right of ax3, with 20% width of ax1
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar2 = plt.colorbar(im2, cax=cax2)
        
        ax3 = self.fig.add_subplot(1,2,2)

        cmap = plt.get_cmap("hot")
        ax3.set_xlabel("pixel")
        ax3.set_xticks([0,99,199,299,399])
        ax3.set_xticklabels([1,100,200,300,400])
        ax3.set_ylabel("normalised counts")
        
        cNorm = colors.Normalize(vmin=np.min(self.smallImage), vmax=np.max(self.smallImage))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        ax3.plot(np.arange(2*extent*2*extent), self.smallImage, "k-", linewidth=0.5)
        i=0

        for element in self.smallImage:
            colorVal = scalarMap.to_rgba(element)
            ax3.plot(i, element, color=colorVal, marker="o")
            i+=1
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, \
                                 top=0.9, wspace=0.4, hspace=0.1)

    
def main():
    parser = optparse.OptionParser("[!] usage: python wx_mpl_display_detections_gui.py\n"+\
                                   "\t -F <data file>\n"+\
                                   "\t -s <data set>\n"+\
                                   "\t -i <info file>\n"+\
                                   "\t -c <classifier file> [optional]\n"+\
                                   "\t -t <threshold [optional, default=0.5]>\n"+\
                                   "\t -p <pooled features file [optional]>")
    
    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")
    parser.add_option("-c", dest="classifierFile", type="string", \
                      help="specify classifier to use")
    parser.add_option("-i", dest="infoFile", type="string", \
                      help="specify information file to use")
    parser.add_option("-t", dest="threshold", type="float", \
                      help="specify decision boundary threshold [optional, default=0.5]")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse (training, test)")
    parser.add_option("-p", dest="poolFile", type="string", \
                      help="specify pooled features file [optional]")
        
    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    infoFile = options.infoFile
    classifierFile = options.classifierFile
    threshold = options.threshold
    dataSet = options.dataSet
    poolFile = options.poolFile
    
    dataSetDict = {}
    
    if dataFile == None or dataSet == None:
    	print parser.usage
        exit(0)
        
    if threshold == None:
        threshold = 0.5
        
    try:
        data = loadData(dataFile)
    except IOError:
        print "[!] Exiting: %s Not Found" % (dataFile)
        exit(0)

    if dataSet == "training":
        try:
            X = data[0]
            y = np.squeeze(data[2])
            files = data[4]
        except IndexError:
            X = data[0]
            y = np.squeeze(data[1])
            files = data[2]
    elif dataSet == "test":
        X = data[1]
        y = np.squeeze(data[3])
        files = data[5]

    infoDict = {}
    if infoFile == None:
       for i,file in enumerate(files):
           infoDict[file.rstrip()] = [file.rstrip()]
    else:
        try:
            for line in open(infoFile,"r").readlines():
                info = line.rstrip().split(",")
                infoDict[info[-1].split("/")[-1].rstrip()] = info
        except IOError:
            print "[!] Exiting: %s Not Found" % (infoFile)
            exit(0)

    if classifierFile != None:
        if poolFile != None:
            try:
                features = sio.loadmat(poolFile)
                pooledFeaturesTrain = features["pooledFeaturesTrain"]
                pooledX = np.transpose(pooledFeaturesTrain, (0,2,3,1))
                numTrainImages = np.shape(pooledX)[3]
                pooledX = np.reshape(pooledX, ((pooledFeaturesTrain.size)/float(numTrainImages), \
                                     numTrainImages), order="F")
                #scaler = preprocessing.MinMaxScaler()
                #scaler.fit(pooledX.T)  # Don't cheat - fit only on training data
                # load pooled feature scaler
                #scaler = mlutils.getMinMaxScaler("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_naturalImages_6x6_signPreserveNorm_pooled5.mat")
                scaler = mlutils.getMinMaxScaler("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stlTrainSubset_whitened_6x6_signPreserveNorm_pooled5.mat")
                pooledX = scaler.transform(pooledX.T)
                if dataSet == "test":
                    pooledFeaturesTest = features["pooledFeaturesTest"]
                    pooledX = np.transpose(pooledFeaturesTest, (0,2,3,1))
                    numTestImages = np.shape(pooledX)[3]
                    pooledX = np.reshape(pooledX, ((pooledFeaturesTest.size)/float(numTestImages), \
                                         numTestImages), order="F")
                    pooledX = scaler.transform(pooledX.T)
            except IOError:
                print "[!] Exiting: %s Not Found" % (poolFile)
                exit(0)
            finally:
                features = None
                pooledFeaturesTrain = None
                pooledFeaturesTest = None
        try :
            true_pos, false_neg, true_neg, false_pos, pred = \
            profileData(classifierFile, pooledX, np.squeeze(y), threshold)
        except NameError:
            true_pos, false_neg, true_neg, false_pos, pred = \
            profileData(classifierFile, X, np.squeeze(y), threshold)


        dataSetDict["true_pos_X"] = X[y==1,:][true_pos,:]
        dataSetDict["true_pos_y"] = y[y==1][true_pos]
        dataSetDict["true_pos_files"] = [str(x).rstrip() for x in files[y==1][true_pos]]

        dataSetDict["false_neg_X"] = X[y==1,:][false_neg,:]
        dataSetDict["false_neg_y"] = y[y==1,:][false_neg,:]
        dataSetDict["false_neg_files"] = [str(x).rstrip() for x in files[y==1,:][false_neg]]

        dataSetDict["real_X"] = X[y==1,:]
        dataSetDict["real_y"] = y[y==1]
        dataSetDict["real_files"] = [str(x).rstrip() for x in files[y==1]]
        try:
            dataSetDict["true_neg_X"] = X[y==0,:][true_neg,:]
            dataSetDict["true_neg_y"] = y[y==0,:][true_neg,:]
            dataSetDict["true_neg_files"] = [str(x).rstrip() for x in files[y==0,:][true_neg]]

            dataSetDict["false_pos_X"] = X[y==0,:][false_pos,:]
            dataSetDict["false_pos_y"] = y[y==0,:][false_pos,:]
            dataSetDict["false_pos_files"] = [str(x).rstrip() for x in files[y==0,:][false_pos]]

            dataSetDict["bogus_X"] = X[y==0,:]
            dataSetDict["bogus_y"] = y[y==0]
            dataSetDict["bogus_files"] = [str(x).rstrip() for x in files[y==0]]
        except IndexError:
            print "[!] No negative examples found."


    dataSetDict["X"] = X
    dataSetDict["y"] = y
    dataSetDict["files"] = [str(x).rstrip() for x in files]
    
    dataSetDict["bogus_X"] = X[y==0,:]
    dataSetDict["bogus_y"] = y[y==0]
    dataSetDict["bogus_files"] = [str(x).rstrip() for x in files[y==0]]
    
    dataSetDict["real_X"] = X[y==1,:]
    dataSetDict["real_y"] = y[y==1]
    dataSetDict["real_files"] = [str(x).rstrip() for x in files[y==1]]
    #print infoDict.keys()
    app = wx.App(False)
    app.frame = mainFrame(dataSetDict, infoDict)
    app.frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
