import pandas as pd
import numpy as np
from bokeh.plotting import ColumnDataSource, figure#, curdoc 
from bokeh.plotting import show as showPlot
from bokeh.io import show, curdoc
from bokeh.models import CustomJS, RadioButtonGroup
from bokeh.layouts import layout
from bokeh.models import HoverTool
LABELS = ["IMPS", "DTA", "REAL RESIUALE", "IMAG RESIDUALS"]

def create_figure():
    if radio_button_group.active == 0:
        plot=figure()
        source_data = ColumnDataSource(data = df_data)
        
        plot.circle(x = df_data.columns[6], y = df_data.columns[7], source = source_data)
        plot.line(x = df_data.columns[8], y = df_data.columns[9], source = source_data)
        TOOLTIPS=[
        ("freq" ,"$freq"),
        ("x", "@x"),
        ("y", "@y"),]
        plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    if radio_button_group.active == 1:
        plot=figure()
        source_data = ColumnDataSource(data = df_dta)
        
        plot.line(x = df_dta.columns[5], y = df_dta.columns[6], source = source_data)
        TOOLTIPS=[
        ("freq" ,"$index"),
        ("x", "@x"),
        ("y", "@y"),]
        plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    return plot

def update_plot(attrname, old, new):
    layout.children[1] = create_figure()

df_data = pd.read_csv("interp_fit.csv")
df_dta = pd.read_csv("dta_tau.csv")
source_dta = ColumnDataSource(data= df_dta)
radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
radio_button_group.js_on_click(CustomJS(code="""
            console.log('radio_button_group: active=' + this.active, this.toString())
                    """))
radio_button_group.on_change('active', update_plot)  
plot = create_figure()

layout = layout([
        [radio_button_group],
        [plot],

    ])
curdoc().add_root(layout)
