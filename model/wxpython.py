import wx
import json
import pathlib

def _scale_bitmap(bitmap, width, height):
    image = wx.ImageFromBitmap(bitmap)
    original_width = image.GetWidth()
    original_height = image.GetHeight()
    
    coeff_by_width = width / original_width
    coeff_by_height = height / original_height

    dst_width = width
    dst_height = original_height * coeff_by_width
    if dst_height > height:
        dst_height = height
        dst_width = coeff_by_height * original_width

    image = image.Scale(int(dst_width),int(dst_height), wx.IMAGE_QUALITY_HIGH)
    result = wx.BitmapFromImage(image)
    return result

class BitmapDisplayPanel(wx.Panel):
    def __init__(self,*args, **kw):
        super(wx.Panel,self).__init__(*args,**kw)

        self.original_bitmap = None
        self.disp_bitmap = None

        self.Bind(wx.EVT_SIZE, self.onSize)
        self.Bind(wx.EVT_PAINT, self.onPaint)

    def _recalculate_display(self, width, height):
        self.disp_bitmap = _scale_bitmap(self.original_bitmap, width, height)
        self.Refresh()

    def setBitmap(self, bitmap):
        self.original_bitmap = bitmap
        width, height = self.GetSize()
        self._recalculate_display(width,height)

    def onPaint(self, event):
        if self.disp_bitmap is None:
            return
        
        dc = wx.PaintDC(self)

        panel_w, panel_h = self.GetSize()
        img_w, img_h = self.disp_bitmap.GetSize()

        dst_x = 0.5 * (panel_w - img_w) 
        dst_y = 0.5 * (panel_h - img_h)

        dc.DrawBitmap(self.disp_bitmap,int(dst_x),int(dst_y))

    def onSize(self, event):
        if self.original_bitmap is not None:
            width, height = event.GetSize()
            self._recalculate_display(width, height)
            

class HelloFrame(wx.Frame):
    #initializer
    def __init__(self, *args, **kw):
        super(HelloFrame,self).__init__(*args,**kw)
        width,height = self.GetSize()

        self.sizer_top_level = wx.BoxSizer(wx.VERTICAL)
        self.sizer_display = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_input = wx.BoxSizer(wx.HORIZONTAL)

        self.counter = 1
        self.progress = {}

        #panel
        self.displayPanel = wx.Panel(self)
        self.imgDisp1 = BitmapDisplayPanel(self.displayPanel)
        self.imgDisp2 = BitmapDisplayPanel(self.displayPanel)

        self.imgDisp1.setBitmap(wx.Bitmap("fox_box.jpg"))
        self.imgDisp2.setBitmap(wx.Bitmap("output\cont1.png"))

        self.sizer_display.Add(self.imgDisp1,1,wx.GROW)
        self.sizer_display.Add(self.imgDisp2,1,wx.GROW)
        self.displayPanel.SetSizer(self.sizer_display)

        self.inputPanel = wx.Panel(self)
        self.textBox = wx.TextCtrl(self.inputPanel,style=wx.TE_PROCESS_ENTER)
        font = self.textBox.GetFont()
        font.PointSize = 15
        self.textBox.SetFont(font)

        self.textBox.Bind(wx.EVT_TEXT_ENTER,self.HandleNext)

        self.button4 = wx.Button(self.inputPanel)
        self.button4.SetLabel("OK")
        self.sizer_input.Add(self.textBox,3,wx.GROW)
        self.sizer_input.Add(self.button4,1,wx.GROW)
        self.inputPanel.SetSizer(self.sizer_input)

        self.button4.Bind(wx.EVT_BUTTON,self.HandleNext)



        self.sizer_top_level.Add(self.displayPanel,3, wx.GROW)
        self.sizer_top_level.Add(self.inputPanel,1,wx.GROW)
        #self.image2 = wx.Bitmap("cont18.png")
        #self.image = scale_bitmap(wx.Bitmap("cont18.png"),width,height)
        #self.text_box = wx.TextCtrl(pnl2, size=(50,-1))

        #self.Bind(wx.EVT_PAINT,self.OnPaint)

        #st = wx.StaticText(pnl,label="Hello World!")
        #font = st.GetFont()
        #font.PointSize +=10
        #font = font.Bold()
        #st.SetFont(font)

        self.SetSizer(self.sizer_top_level)
        self.Show()

    def save_progress(self):
        with open("progress.json", 'w') as f:
            f.write(json.dumps(self.progress))

    def HandleNext(self,event):
        if self.textBox.GetNumberOfLines() != 1:
            wx.MessageBox("invalid input, please input one line", "error", wx.OK | wx.ICON_ERROR)
            return
        if self.textBox.GetLineLength(0) >= 1:
            text = self.textBox.GetLineText(0).strip()
        else:
            text = ""
        self.textBox.Clear()

        current_file_name = f"cont{self.counter}.png"
        self.progress[current_file_name] = text
        self.save_progress()

        self.counter = self.counter+1
        self.imgDisp2.setBitmap(wx.Bitmap(str(pathlib.Path("output") / f"cont{self.counter}.png")))

    def OnExit(self,event):
        self.Close(True)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)

        #width1 = self.image1.GetWidth()
        #height1 = self.image1.GetHeight()

        #width2 = self.image2.GetWidth()
        #height2 = self.image2.GetHeight()

        dc.DrawBitmap(self.image,0,0)
        #dc.DrawBitmap(self.image2,width1,0)
        
if __name__ == '__main__':
    app = wx.App()
    frm = HelloFrame(None,title='Image Labelling')
    app.MainLoop()



