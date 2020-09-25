#-----------------------------------------------------------------------------
# Copyright (c) 2012 - 2020, Anaconda, Inc., and Bokeh Contributors.
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#-----------------------------------------------------------------------------
# see https://raw.githubusercontent.com/bokeh/bokeh/ffdd1114e6aace02bb6c61748390e9b62522a8d9/bokeh/themes/_ggplot.py
# see https://github.com/bokeh/bokeh/pull/10150
from bokeh.themes import Theme
json = {
    "attrs": {
        "Figure" : {
            "background_fill_color": "#E5E5E5",
            "border_fill_color": "#FFFFFF",
            "outline_line_color": "#000000",
            "outline_line_alpha": 0.25
        },

        "Grid": {
            "grid_line_color": "#FFFFFF",
            "grid_line_alpha": 1
        },

        "Axis": {
            "major_tick_line_alpha": 0.3,
            "major_tick_line_color": "#000000",

            "minor_tick_line_alpha": 0.4,
            "minor_tick_line_color": "#000000",

            "axis_line_alpha": 1,
            "axis_line_color": "#000000",

            "major_label_text_color": "#000000",
            "major_label_text_font": "Helvetica",
            "major_label_text_font_size": "1.025em",

            "axis_label_standoff": 10,
            "axis_label_text_color": "#000000",
            "axis_label_text_font": "Helvetica",
            "axis_label_text_font_size": "1.25em",
            "axis_label_text_font_style": "normal"
        },

        "Legend": {
            "spacing": 8,
            "glyph_width": 15,

            "label_standoff": 8,
            "label_text_color": "#000000",
            "label_text_font": "Arial",
            "label_text_font_size": "0.95em",

            "border_line_alpha": 1,
            "background_fill_alpha": 0.25,
            "background_fill_color": "#000000"
        },

        "ColorBar": {
            "title_text_color": "#E0E0E0",
            "title_text_font": "Helvetica",
            "title_text_font_size": "1.025em",
            "title_text_font_style": "normal",

            "major_label_text_color": "#E0E0E0",
            "major_label_text_font": "Arial",
            "major_label_text_font_size": "1.025em",

            "background_fill_color": "#000000",
            "major_tick_line_alpha": 0,
            "bar_line_alpha": 0
        },

        "Title": {
            "text_color": "#000000",
            "text_font": "Helvetica",
            "text_font_size": "1.10em"
        }
    }
}
ggplot_theme = Theme(json=json)
