import ipywidgets as widgets
from IPython.display import display

def set_text(string_name, string_default_val, show = True):
    widget_text = widgets.Text(
        value = '',
        placeholder = string_default_val,
        description= string_name,
        disable = False,
        layout=widgets.Layout(flex='1 1 auto', width='auto'),
        style = {'description_width': 'initial'})
    
    if show:
        display(widget_text)
    return widget_text

def set_Int(string_name, default_value, show = True):
    widget_int = widgets.IntText(
        value = default_value,
        description = string_name,
        disabled=False,
        layout=widgets.Layout(flex='1 1 auto', width='auto'),
        style = {'description_width': 'initial'})    
    
    #a = widgets.HBox([widgets.Label(string_name, layout=widgets.Layout( width='200px')), widget_int])
    if show:
        display(widget_int)
    return widget_int 


def set_checkbox(string_name, default_value, show = True):
    widget_checkbox = widgets.Checkbox(
        value=default_value,
        description= string_name,
        disabled=False,
    )
    if show:
        display(widget_checkbox)
    return widget_checkbox

def set_IntRangeSlider(string_name, low_val, high_val, min_val, max_val, show = True ):
    widget_IntRangeSlider = widgets.IntRangeSlider(
        value=[low_val, high_val],
        min= min_val,
        max= max_val,
        step=1,
        description=string_name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        #layout=widgets.Layout(flex='1 1 auto', width='auto')
    )
    if show:
        display(widget_IntRangeSlider)
    return widget_IntRangeSlider

def set_IntSlider(string_name, initial_val, min_val, max_val, show = True ):
    widget_intSlider = widgets.widgets.IntSlider(
        value = initial_val,
        min = min_val,
        max = max_val,
        step=1,
        description= string_name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(flex='1 1 auto', width='auto')
    )
    if show:
        display(widget_intSlider)
    return widget_intSlider

def set_FloatRangeSlider(string_name, low_val, high_val, min_val, max_val, show = True ):
    widget_intSlider = widgets.FloatRangeSlider(
        value=[low_val, high_val],
        min= min_val,
        max= max_val,
        step= 0.1,
        description=string_name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(flex='1 1 auto', width='auto')
    )
    if show:
        display(widget_intSlider)
    return widget_intSlider

def set_dropdown(string_name, default_options):
    widget_dropdown = widgets.Dropdown(
        options = default_options,
        #value= default_options[0],
        description = string_name,
        disabled = False,
        style = {'description_width': 'initial'}
        #indent=False,
    )
    display(widget_dropdown)
    return widget_dropdown

def set_Float_Bounded(string_name, default_value, min_val, max_val, step_val):
    widget_float_bounded = widgets.BoundedFloatText(
        description = string_name,
        value = default_value,
        min = min_val,
        max = max_val,
        step = step_val, 
        style = {'description_width': 'initial'}) 
    return widget_float_bounded 