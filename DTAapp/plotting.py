import pandas as pd
import numpy as np
from bokeh.plotting import ColumnDataSource, figure#, curdoc 
from bokeh.plotting import show as showPlot
from bokeh.io import show, curdoc
from bokeh.models import CustomJS, RadioButtonGroup
from bokeh.layouts import layout
from bokeh.models import HoverTool
from django.conf import settings
from bokeh.layouts import row, column
from bokeh.models.axes import LinearAxis
from bokeh.palettes import RdYlGn

LABELS = ["IMPS","REAL RESIDUALS", "IMAG RESIDUALS"]
class myfig(object):
    def __init__(self, df_data, df_dta):
        self.df_data = df_data
        self.df_dta = df_dta

        self.radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
        self.source = None
        self.plot = self.create_grid()
        self.radio_button_group.js_on_change('active', CustomJS(args=
                dict(s=self.source,
                    p = self.plot.children[0],
                   ),
                   code="""
console.log('radio_button_group: active=' + this.active, this.toString());
let s1 = {'index': [0,1,2,3],'1':[3,4,5,6],'2':[1,3,2,4],'3':[5,4,7,6],'4':[5,5,5,9],'5': [-4,-5,5,-2]} ;
s.data = s1;
switch(this.active){
case 0:
console.log('the first');
break;
case 1:
console.log('the second');
break;
case 2:
console.log('the third');
break;
};
s.change.emit();
                     """))
        self.layout = layout([
               [self.radio_button_group],
                [self.plot],
           ])

    def create_grid(self):
        plot_data=figure(title= "IMPS", x_axis_label="Y'",  y_axis_label="Y''")
        source_data = ColumnDataSource(data = self.df_data)
        vec_size = np.arange(0,len(self.df_data.columns), 1,dtype=int)
        data_to_plot = vec_size.reshape(int(len(self.df_data.columns)/7), 7)
        palettes_ = RdYlGn[data_to_plot.shape[0]] if data_to_plot.shape[0]>2 and data_to_plot.shape[0]<12 else RdYlGn[11] 
        self.source = source_data
        for number, fix_voltage in enumerate(data_to_plot):
            plot_data.circle(x = self.df_data.columns[fix_voltage[1]], y = self.df_data.columns[fix_voltage[2]], source = self.source,color=palettes_[number])
            plot_data.line(x = self.df_data.columns[fix_voltage[3]], y = self.df_data.columns[fix_voltage[4]], source = self.source, color = palettes_[number], line_dash = "dashed")
        frequency_index ="@{" +list(source_data.data.keys())[1]+"}"
        x_index = "@{" +list(source_data.data.keys())[2]+"}"
        y_index = "@{" +list(source_data.data.keys())[3]+"}"
        TOOLTIPS=[
            ("freq" ,frequency_index),
            ("x", "$x"),
            ("y", "$y"),]
        plot_data.add_tools(HoverTool(tooltips=TOOLTIPS))
        plot_dta = figure(title="DTA result",  x_axis_type="log",x_axis_label="Time (s)",  y_axis_label="DTA (V^-1)")
        source_dta = ColumnDataSource(data = self.df_dta)

        vec_size = np.arange(0,len(self.df_dta.columns), 1,dtype=int)
        data_to_plot = vec_size.reshape(int(len(self.df_dta.columns)/2),2)
        for number, fix_voltage in enumerate(data_to_plot):
            plot_dta.line(x = self.df_dta.columns[fix_voltage[0]], y = self.df_dta.columns[fix_voltage[1]], source = source_dta,color = palettes_[number],)
        freq ="@" + self.df_data.columns[0] 
        TOOLTIPS=[
            ("tau", "$x"),
            ("DTA", "$y"),]
        plot_dta.add_tools(HoverTool(tooltips=TOOLTIPS))
        grid = row(plot_data, plot_dta)
        return grid

    def update_plot(self, attrname, old, new):
        self.layout.children[1] = self.create_figure()
if __name__ =='__main__':
    from bokeh.plotting import show 
    mydata = pd.DataFrame({"1": [4, 5, 6, 7], "2": [7, 8, 9, 10], "3": [1, 2, 1, 2], "4":
                           [9, 8, 7, 6], "5": [8, 8, 8, 8], "6": [9, 8, 7, 6],
                          "7":[34, 54, 12, 3], "8": [4, 5, 6, 7], "9": [1, 1,
                                                                        1, 2],
                          "10": [12, 22, 31, 22], "11":
                           [9, 38, 47, 16], "12": [48, 48, 48, 48], "13": [19,
                                                                           18,
                                                                           17,
                                                                           16],
                          "14":[4, 4, 2, 3]})
    yourdata = pd.DataFrame({"1":[1,2,3,4], "2": [1,2,3,4]})
    fig = myfig(mydata, yourdata)
    show(fig.layout)


