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
LABELS = ["IMPS","REAL RESIDUALS", "IMAG RESIDUALS"]
class myfig(object):
    def __init__(self, df_data, df_dta):
        self.df_data = df_data
        self.df_dta = df_dta

        self.radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
        self.radio_button_group.js_on_click(CustomJS(code="""
                    console.log('radio_button_group: active=' + this.active, this.toString())
                            """))
       
        #self.update_plot)  
        self.source= None
        self.plot = self.create_grid()

        self.layout = layout([
               [self.radio_button_group],
                [self.plot],
           ])
        #if self.df_data != None and self.df_dta != None:
        self.update = CustomJS(args=dict(
        p=self.layout.children[1],
        df_dtax = self.df_dta[self.df_dta.columns[0]],
        df_dtay =self.df_dta[self.df_dta.columns[1]],   
        df_datax = self.df_data[ self.df_data.columns[0]], 
        df_datay = self.df_data[ self.df_data.columns[1]], 
        source = self.source),
        code = """    
// Javascript code for the callback
const radio = 1;
var plot = p;
const x = Bokeh.LinAlg.linspace(-0.5, 20.5, 10);
const y = x.map(function (v) { return v * 0.5 + 3.0; });
var source = "None"
const y2 = x.map(function (v) { return v * v + 3.0; });

switch (radio){
case 0: 
source = new Bokeh.ColumnDataSource({ data: { x: x, y: y2 } });
break;
case 1:
source = new  Bokeh.ColumnDataSource({ data: { x: x, y: y } });
break;
} 
plot = new Bokeh.Plot({
title: "BokehJS Plot",
width: 400,
height: 400,
background_fill_color: "#F2F2F7"
});

const line = new Bokeh.Line({
x: { field: "x" },
y: { field: "y" },
line_color: "#666699",
line_width: 2
});
plot.add_glyph(line, source);
p = [plot]
""")
        self.radio_button_group.js_on_change('active', self.update)

    def create_grid(self):
        
        plot_data=figure(title= "IMPS", x_axis_label="Y'",  y_axis_label="Y''")
        source_data = ColumnDataSource(data = self.df_data)
        vec_size = np.arange(0,len(self.df_data.columns), 1,dtype=int)
        #columns = int(len(self.df_data.columns)/(len(self.df_data.columns)/5)))
        data_to_plot = vec_size.reshape(int(len(self.df_data.columns)/5),5)
        for fix_voltage in data_to_plot:
            plot_data.circle(x = self.df_data.columns[fix_voltage[1]], y = self.df_data.columns[fix_voltage[2]], source = source_data)
            plot_data.line(x = self.df_data.columns[fix_voltage[3]], y = self.df_data.columns[fix_voltage[4]], source = source_data)
        #plot_data.add_layout(LinearAxis(y_range_name="Y''"), 'left')
        #plot_data.add_layout(LinearAxis(x_range_name="Y'"), 'below')
        TOOLTIPS=[
            ("freq" ,"@freq"),
            ("x", "$x"),
            ("y", "$y"),]
        plot_data.add_tools(HoverTool(tooltips=TOOLTIPS))
        plot_dta = figure(title="DTA result",  x_axis_type="log",x_axis_label="Time (s)",  y_axis_label="DTA (V^-1)")
        source_dta = ColumnDataSource(data = self.df_dta)
            
        vec_size = np.arange(0,len(self.df_dta.columns), 1,dtype=int)
        data_to_plot = vec_size.reshape(int(len(self.df_dta.columns)/2),2)
        for fix_voltage in data_to_plot:
            plot_dta.line(x = self.df_dta.columns[fix_voltage[0]], y = self.df_dta.columns[fix_voltage[1]], source = source_dta)
        TOOLTIPS=[
            ("freq" ,"@tau"),
            ("x", "$x"),
            ("y", "$y"),]
        plot_dta.add_tools(HoverTool(tooltips=TOOLTIPS))
        grid = row(plot_data, plot_dta)
        return grid

    def update_plot(self, attrname, old, new):
        self.layout.children[1] = self.create_figure()

   #curdoc().add_root(layout)"""
"""
switch (radio){
case 0:
const source = new Bokeh.ColumnDataSource({data=dict( x= df_datax, y=df_datay ));
break;
case 1:
const source = new Bokeh.ColumnDataSource({data=dict( x= df_dtax, y=df_dtay ));
break;
};
plot = new Bokeh.Plot({
title: "DTAplot",
width: 400,
height: 400,
background_fill_color: "#F2F2F7"
});
const circle = new Bokeh.Line({
x:{field: "x"},
y: {field:"y"}
});
plot.add_glyph(circle, source);
p=plot;
p.change.emit();
        """#
