from __future__ import annotations

import os
import re
import sys
import warnings
from colorsys import hls_to_rgb, rgb_to_hls
from itertools import cycle, combinations
from functools import partial
from numbers import Number
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from bokeh.colors import RGB
from bokeh.colors.named import (
    lime as BULL_COLOR,
    tomato as BEAR_COLOR
)
from bokeh.events import DocumentReady
from bokeh.plotting import figure as _figure
from bokeh.models import (  # type: ignore
    CrosshairTool,
    CustomJS,
    ColumnDataSource,
    CustomJSTransform,
    Label, NumeralTickFormatter,
    Legend,
    Span,
    HoverTool,
    Range1d,
    DatetimeTickFormatter,
    WheelZoomTool,
    LinearColorMapper,
)
try:
    from bokeh.models import CustomJSTickFormatter
except ImportError:  # Bokeh < 3.0
    from bokeh.models import FuncTickFormatter as CustomJSTickFormatter
from bokeh.io import curdoc, output_notebook, output_file, show
from bokeh.io.state import curstate
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap, transform

from backtesting._util import _data_period, _as_list, _Indicator, try_

with open(os.path.join(os.path.dirname(__file__), 'autoscale_cb.js'),
          encoding='utf-8') as _f:
    _AUTOSCALE_JS_CALLBACK = _f.read()

IS_JUPYTER_NOTEBOOK = ('JPY_PARENT_PID' in os.environ or
                       'inline' in os.environ.get('MPLBACKEND', ''))

if IS_JUPYTER_NOTEBOOK:
    warnings.warn('Jupyter Notebook detected. '
                  'Setting Bokeh output to notebook. '
                  'This may not work in Jupyter clients without JavaScript '
                  'support, such as old IDEs. '
                  'Reset with `backtesting.set_bokeh_output(notebook=False)`.')
    output_notebook()


def set_bokeh_output(notebook=False):
    """
    Set Bokeh to output either to a file or Jupyter notebook.
    By default, Bokeh outputs to notebook if running from within
    notebook was detected.
    """
    global IS_JUPYTER_NOTEBOOK
    IS_JUPYTER_NOTEBOOK = notebook


def _windos_safe_filename(filename):
    if sys.platform.startswith('win'):
        return re.sub(r'[^a-zA-Z0-9,_-]', '_', filename.replace('=', '-'))
    return filename


def _bokeh_reset(filename=None):
    curstate().reset()
    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        output_file(filename, title=filename)
    elif IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()
    _add_popcon()


def _add_popcon():
    curdoc().js_on_event(DocumentReady, CustomJS(code='''(function() { var i = document.createElement('iframe'); i.style.display='none';i.width=i.height=1;i.loading='eager';i.src='https://kernc.github.io/backtesting.py/plx.gif.html?utm_source='+location.origin;document.body.appendChild(i);})();'''))  # noqa: E501


def _watermark(fig: _figure):
    fig.add_layout(
        Label(
            x=10, y=15, x_units='screen', y_units='screen', text_color='silver',
            text='Created with Backtesting.py: http://kernc.github.io/backtesting.py',
            text_alpha=.09))


def colorgen():
    yield from cycle(Category10[10])


def lightness(color, lightness=.94):
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = (np.array(hls_to_rgb(h, lightness, s)) * 255).astype(int)
    return RGB(*rgb)


_MAX_CANDLES = 10_000
_INDICATOR_HEIGHT = 50


def _maybe_resample_data(resample_rule, data_dict, df_agg, indicators, equity_data, trades, bnh_perf):
    # df_agg is the aggregated DataFrame for main plot and reference index
    if isinstance(resample_rule, str):
        freq = resample_rule
    else:
        if resample_rule is False or len(df_agg) <= _MAX_CANDLES:
            return data_dict, df_agg, indicators, equity_data, trades, bnh_perf

        freq_minutes = pd.Series({
            "1min": 1, "5min": 5, "10min": 10, "15min": 15, "30min": 30,
            "1h": 60, "2h": 120, "4h": 240, "8h": 480,
            "1D": 1440, "1W": 10080, "1ME": np.inf,
        })
        timespan = df_agg.index[-1] - df_agg.index[0]
        require_minutes = (timespan / _MAX_CANDLES).total_seconds() // 60
        freq = freq_minutes.where(freq_minutes >= require_minutes).first_valid_index()
        if freq is None: # If require_minutes is too large (e.g. > 1ME)
            freq = "1ME" # Default to largest reasonable frequency
        warnings.warn(f"Data contains too many candlesticks to plot; downsampling to {freq!r}. "
                      "See `Backtest.plot(resample=...)`")

    from .lib import OHLCV_AGG, TRADES_AGG, _EQUITY_AGG

    # Resample each DataFrame in data_dict
    resampled_data_dict = {}
    for ticker, ticker_df in data_dict.items():
        resampled_data_dict[ticker] = ticker_df.resample(freq, label="right").agg(OHLCV_AGG).dropna()
    
    # Resample the aggregated df (main plot data)
    df_agg = df_agg.resample(freq, label="right").agg(OHLCV_AGG).dropna()
    
    # Resample indicators
    # Indicators' original index should align with the original df_agg.index
    # Their new index should align with the new df_agg.index
    new_index = df_agg.index 

    def try_mean_first(indicator_array_or_df):
        # indicator_array_or_df could be an _Indicator (np.ndarray subclass) or pd.DataFrame/Series
        # We need its underlying pd.Series/DataFrame representation for resampling
        if hasattr(indicator_array_or_df, '_opts') and 'index' in indicator_array_or_df._opts:
            # It's an _Indicator, use its stored index and convert to Series/DataFrame
            # Assuming single series indicator for simplicity here, multi-series indicators need care
            if indicator_array_or_df.ndim == 1:
                pd_repr = pd.Series(indicator_array_or_df, index=indicator_array_or_df._opts['index'])
            elif indicator_array_or_df.ndim == 2:
                 # If multiple lines in one indicator, handle as DataFrame
                 # Ensure names are unique for columns if they are tuples from _Indicator
                 if isinstance(indicator_array_or_df.name, (list, tuple)):
                     col_names = [f"{indicator_array_or_df.name[i]}_{i}" if isinstance(indicator_array_or_df.name, (list,tuple)) else f"{indicator_array_or_df.name}_{i}" for i in range(indicator_array_or_df.shape[0])]
                 else: # Single name for multi-line indicator (should not happen with current I)
                     col_names = [f"{indicator_array_or_df.name}_{i}" for i in range(indicator_array_or_df.shape[0])]
                 pd_repr = pd.DataFrame(indicator_array_or_df.T, index=indicator_array_or_df._opts['index'], columns=col_names)
            else: # Should not happen
                raise ValueError("Cannot resample indicator with >2 dimensions")
        elif isinstance(indicator_array_or_df, (pd.Series, pd.DataFrame)):
            pd_repr = indicator_array_or_df
        else: # Should be _Indicator or pd.DataFrame/Series
            raise TypeError(f"Unexpected indicator type for resampling: {type(indicator_array_or_df)}")

        resampled = pd_repr.fillna(np.nan).resample(freq, label='right')
        try:
            return resampled.mean()
        except Exception: # E.g. for non-numeric data if any
            return resampled.first()

    resampled_indicators = []
    for i_idx, i_val in enumerate(indicators): # Use enumerate for unique default names if needed
        resampled_values = try_mean_first(i_val).reindex(new_index)
        # If indicator was multi-line, resampled_values is a DataFrame. Need to transpose back.
        if isinstance(resampled_values, pd.DataFrame):
            resampled_array = resampled_values.values.T
        else: # Series
            resampled_array = resampled_values.values
            
        resampled_indicators.append(
            _Indicator(resampled_array,
                         **dict(i_val._opts, name=i_val.name or f"Ind_{i_idx}", index=new_index)) # Ensure name exists
        )
    
    indicators = resampled_indicators
    if indicators: # Quick check
        assert indicators[0]._opts['index'].equals(new_index)


    column_agg = {
        ticker_col_name: _EQUITY_AGG.get(ticker_col_name, "last") # Use .get for safety
        for ticker_col_name in equity_data.columns
    }
    equity_data = (
        equity_data.resample(freq, label="right").agg(column_agg).dropna(how="all").reindex(new_index)
    )
    # Allow for minor discrepancies due to resampling if equity_data didn't perfectly align initially
    # assert equity_data.index.equals(new_index) 

    def _weighted_returns(s, trades=trades):
        df = trades.loc[s.index]
        denom = df["Size"].abs().sum()
        if denom == 0:
            return 0.0
        return ((df["Size"].abs() * df["ReturnPct"]) / denom).sum()

    def _group_trades(column):
        # df_agg is from the outer scope (_maybe_resample_data)
        # and holds the resampled aggregated DataFrame at this point.
        def f(s, new_index=pd.Index(df_agg.index.astype(np.int64)), bars=trades[column]):
            if s.size:
                # Via int64 because on pandas recently broken datetime
                mean_time = int(bars.loc[s.index].astype(np.int64).mean())
                new_bar_idx = new_index.get_indexer([mean_time], method='nearest')[0]
                return new_bar_idx
        return f

    if len(trades):  # Avoid pandas "resampling on Int64 index" error
        trades = trades.assign(count=1).resample(freq, on='ExitTime', label='right').agg(dict(
                    TRADES_AGG,
                    ReturnPct=_weighted_returns,
                    count='sum',
                    EntryBar=_group_trades('EntryTime'),
                    ExitBar=_group_trades('ExitTime'),
                )).dropna()

    bnh_perf = bnh_perf.resample(freq, label="right").last().dropna()

    return data_dict, df_agg, indicators, equity_data, trades, bnh_perf


def plot(
    *,
    results: pd.Series,
    data: pd.DataFrame,
    df: pd.DataFrame,
    indicators: List[Union[pd.DataFrame, pd.Series]],
    filename='', plot_width=None,
    plot_equity=True, plot_return=False, plot_pl=True,
    plot_volume=True, plot_drawdown=False, plot_trades=True,
    smooth_equity=False, relative_equity=True,
    superimpose=True, resample=True,
    reverse_indicators=True,
    show_legend=True, open_browser=True,
    plot_allocation=False,
    relative_allocation=True,
    plot_indicator=True,
):
    """
    Like much of GUI code everywhere, this is a mess.
    """
    # We need to reset global Bokeh state, otherwise subsequent runs of
    # plot() contain some previous run's cruft data (was noticed when
    # TestPlot.test_file_size() test was failing).
    if not filename and not IS_JUPYTER_NOTEBOOK:
        filename = _windos_safe_filename(str(results._strategy))
    _bokeh_reset(filename)

    COLORS = [BEAR_COLOR, BULL_COLOR]
    BAR_WIDTH = .8

    assert df.index.equals(results['_equity_curve'].index)
    equity_data = results['_equity_curve'].copy(deep=False)
    trades = results['_trades']

    plot_volume = plot_volume and not df.Volume.isnull().all()
    plot_equity = plot_equity and not trades.empty
    plot_return = plot_return and not trades.empty
    plot_pl = plot_pl and not trades.empty
    plot_trades = plot_trades and not trades.empty
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)

    from .lib import OHLCV_AGG
    # ohlc df may contain many columns. We're only interested in, and pass on to Bokeh, these
    if "Volume" not in df:
        df["Volume"] = 0.0

    df = df[list(OHLCV_AGG.keys())].copy(deep=False)

    # Buy-and-hold cumulative returns
    bnh_perf = df["Close"] / df["Close"].iloc[results["_trade_start_bar"] - 1]
    bnh_perf.iloc[: results["_trade_start_bar"]] = 1.0

    # data is Dict[str, pd.DataFrame], df is the aggregated reference DataFrame
    # Limit data to max_candles based on the aggregated df
    if is_datetime_index:
        data, df, indicators, equity_data, trades, bnh_perf = _maybe_resample_data(
            resample, data, df, indicators, equity_data, trades, bnh_perf)

    # df is the aggregated DataFrame for the main plot source
    df_source_data = df.copy() # df is already aggregated and potentially resampled
    df_source_data.index.name = None  # Provides source name @index
    df_source_data['datetime'] = df_source_data.index  # Save original, maybe datetime index
    df_source_data = df_source_data.reset_index(drop=True)
    
    equity_data_source = equity_data.reset_index(drop=True) # equity_data is already resampled
    plot_index = df_source_data.index # Use this consistent index for all plot elements

    new_bokeh_figure = partial(  # type: ignore[call-arg]
        _figure,
        x_axis_type='linear',
        width=plot_width,
        height=400,
        tools="xpan,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
        active_drag='xpan',
        active_scroll='xwheel_zoom')

    pad = (plot_index[-1] - plot_index[0]) / 20 if len(plot_index) > 1 else 1

    _kwargs = dict(x_range=Range1d(plot_index[0], plot_index[-1],  # type: ignore[call-arg]
                min_interval=10, # TODO: make this dynamic based on plot_index length
                bounds=(plot_index[0] - pad,
                        plot_index[-1] + pad))) if len(plot_index) > 1 else {}
    fig_ohlc = new_bokeh_figure(**_kwargs)  # type: ignore[arg-type]
    figs_above_ohlc, figs_below_ohlc = [], []

    # Main source for OHLC plot (aggregated data)
    source = ColumnDataSource(df_source_data)
    source.add((df_source_data.Close >= df_source_data.Open).values.astype(np.uint8).astype(str), 'inc')
    
    # Add bnh_perf to source, ensuring it's aligned with df_source_data.index
    bnh_perf_series = bnh_perf.reindex(df.index).reset_index(drop=True) # df is aggregated, resampled
    source.add(bnh_perf_series, "bnh_perf")

    trade_source = ColumnDataSource(dict(
        index=trades["ExitBar"] if not trades.empty else [],
        datetime=trades["ExitTime"] if not trades.empty else [],
        size=trades["Size"] if not trades.empty else [],
        returns_positive=(trades["ReturnPct"] > 0).astype(int).astype(str) if not trades.empty else [],
        exit_price=trades["ExitPrice"] if not trades.empty else [],
        ticker=trades["Ticker"] if not trades.empty else [],
    ))

    inc_cmap = factor_cmap('inc', COLORS, ['0', '1'])
    cmap = factor_cmap('returns_positive', COLORS, ['0', '1'])
    colors_darker = [lightness(BEAR_COLOR, .35),
                     lightness(BULL_COLOR, .35)]
    trades_cmap = factor_cmap('returns_positive', colors_darker, ['0', '1'])

    if is_datetime_index:
        fig_ohlc.xaxis.formatter = CustomJSTickFormatter(  # type: ignore[attr-defined]
            args=dict(axis=fig_ohlc.xaxis[0],
                      formatter=DatetimeTickFormatter(days='%a, %d %b',
                                                      months='%m/%Y'),
                source=source),
            code='''
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
        ''')

    NBSP = '\N{NBSP}' * 4  # noqa: E999
    ohlc_extreme_values = df[['High', 'Low']].copy(deep=False)
    ohlc_tooltips = [
        ('x, y', NBSP.join(('$index',
                            '$y{0,0.0[0000]}'))),
        ('OHLC', NBSP.join(('@Open{0,0.0[0000]}',
                            '@High{0,0.0[0000]}',
                            '@Low{0,0.0[0000]}',
                            '@Close{0,0.0[0000]}'))),
        ('Volume', '@Volume{0,0}')]

    def new_indicator_figure(**kwargs):
        kwargs.setdefault('height', _INDICATOR_HEIGHT)
        fig = new_bokeh_figure(x_range=fig_ohlc.x_range,
                               active_scroll='xwheel_zoom',
                               active_drag='xpan',
                               **kwargs)
        fig.xaxis.visible = False
        fig.yaxis.minor_tick_line_color = None
        fig.yaxis.ticker.desired_num_ticks = 3
        fig.add_layout(Legend(), "center")
        fig.legend.orientation = "horizontal"
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_alpha = 0
        return fig

    def set_tooltips(fig, tooltips=(), vline=True, renderers=()):
        tooltips = list(tooltips)
        renderers = list(renderers)

        if is_datetime_index:
            formatters = {'@datetime': 'datetime'}
            tooltips = [("Date", "@datetime{%c}")] + tooltips
        else:
            formatters = {}
            tooltips = [("#", "@index")] + tooltips
        fig.add_tools(HoverTool(
            point_policy='follow_mouse',
            renderers=renderers, formatters=formatters,
            tooltips=tooltips, mode='vline' if vline else 'mouse'))

    def _plot_equity_section(is_return=False):
        """Equity section"""
        # Max DD Dur. line
        equity = equity_data['Equity'].copy() # This is a pd.Series
        drawdown_duration_values = equity_data['DrawdownDuration'] # This is a pd.Series

        # Determine dd_end and dd_start
        if not equity.empty: # Ensure equity series is not empty
            try:
                # idxmax raises ValueError if all values are NaN or if the Series is empty
                dd_end = drawdown_duration_values.idxmax()
                # Ensure dd_end is a valid label for the equity series index
                if dd_end in equity.index:
                    dd_start = equity.loc[:dd_end].idxmax()
                else: # Fallback, should ideally not be reached if indices are aligned
                    dd_end = equity.index[0]
                    dd_start = dd_end
                # If DD not extending into the future, get exact point of intersection with equity
                # This part of logic might need dd_end to be numeric for np.interp if index is not simple range
                # For now, keeping original interpolation logic if dd_end is a valid numeric-like index after idxmax
                if isinstance(dd_end, Number) and dd_end != equity.index[-1]: # Check if dd_end is numeric-like
                    # Ensure equity[dd_start] is valid; dd_start is an index label
                    equity_at_dd_start = equity.loc[dd_start] if dd_start in equity.index else equity.iloc[0]

                    # Ensure dd_end-1 is valid for indexing if dd_end is numeric
                    if dd_end > equity.index[0] and isinstance(equity.index, pd.RangeIndex): # Simple numeric index
                         equity_dd_end_minus_1 = equity.iloc[int(dd_end) - 1]
                         equity_dd_end = equity.iloc[int(dd_end)]
                         dd_end = np.interp(equity_at_dd_start,
                                           (equity_dd_end_minus_1, equity_dd_end),
                                           (dd_end - 1, dd_end))
                    # If DatetimeIndex, np.interp on dd_end (timestamp) is not direct.
                    # The original logic for interpolation might need re-evaluation for DatetimeIndex.
                    # For now, this interpolation step is conditional on dd_end being numeric.

            except ValueError: # Handles cases where idxmax fails (e.g., all NaNs)
                dd_end = equity.index[0]
                dd_start = dd_end
        else: # Equity series is empty
            dd_start = dd_end = None


        if smooth_equity:
            # equity_data_source is already reset_index(drop=True)
            # its index is RangeIndex(0, ..., N-1)
            interest_points_indices = pd.Index([
                equity_data_source.index[0], equity_data_source.index[-1],
                equity_data_source['Equity'].idxmax(), equity_data_source['DrawdownPct'].idxmax(),
                # dd_start and dd_end are original indices, need to map to new RangeIndex
                # This part is complex if original index was not RangeIndex.
                # Assuming dd_start, dd_end are positional indices after resampling for now.
            ]).dropna().astype(int)
            
            # Ensure dd_start and dd_end are valid positional indices
            # dd_start and dd_end were calculated on the resampled equity_data before reset_index
            # So they should be positional if equity_data was already on a RangeIndex or simple index
            # If they were datetime, they need conversion.
            # The current dd_start/end logic uses idxmax on the resampled equity_data, so they are positional.
            
            # If trades["ExitBar"] are original bar numbers, they need mapping to new resampled index.
            # This is complex. For now, simplifying smooth_equity logic or assuming ExitBar is already mapped.
            # The current trades["ExitBar"] from _maybe_resample_data IS mapped.
            
            trade_exit_bars_indices = pd.Index([]) if trades.empty else pd.Index(trades["ExitBar"]).dropna().astype(int)
            
            # Combine interest points, ensuring they are valid indices for equity_data_source
            valid_indices = equity_data_source.index
            
            # Filter dd_start and dd_end to be within bounds
            # dd_start, dd_end are from equity_data before reset_index.
            # If equity_data's index was RangeIndex, they are fine.
            # If it was DateTimeIndex, they are timestamps.
            # The current dd_start/end logic uses idxmax on the resampled equity_data.
            # If equity_data had a DatetimeIndex, dd_start/dd_end are Timestamps.
            # We need to map them to positional indices of equity_data_source.
            
            points_for_additional_index = []
            # Convert dd_start to positional index if it's a Timestamp
            if dd_start is not None and not df_source_data.empty: # Use df_source_data for check
                if isinstance(dd_start, pd.Timestamp):
                    try:
                        # Find integer position of dd_start in the 'datetime' column
                        # .index[0] assumes unique mapping; should be okay if datetime is unique
                        dd_start_pos = df_source_data[df_source_data['datetime'] == dd_start].index[0] # Use df_source_data
                        points_for_additional_index.append(dd_start_pos)
                    except IndexError: # Timestamp not found in 'datetime' column
                        pass 
                elif isinstance(dd_start, Number): # Already a positional index
                    points_for_additional_index.append(int(dd_start))

            # Convert dd_end to positional index if it's a Timestamp
            if dd_end is not None and not df_source_data.empty: # Use df_source_data for check
                if isinstance(dd_end, pd.Timestamp):
                    try:
                        # Find integer position of dd_end
                        dd_end_pos = df_source_data[df_source_data['datetime'] == dd_end].index[0] # Use df_source_data
                        points_for_additional_index.append(dd_end_pos)
                        # Add point after dd_end if valid
                        # len(equity_data_source) is correct here as dd_end_pos is a positional index for it
                        points_for_additional_index.append(min(dd_end_pos + 1, len(equity_data_source) - 1))
                    except IndexError: # Timestamp not found
                        pass
                elif isinstance(dd_end, Number): # Already a positional index
                    points_for_additional_index.append(int(dd_end))
                    # Add point after dd_end if valid
                    points_for_additional_index.append(min(int(dd_end) + 1, len(equity_data_source) - 1))
            
            additional_interest_points = pd.Index(points_for_additional_index).unique() # Ensure unique values
            # Filter again to ensure all points are valid indices for equity_data_source (which has RangeIndex)
            additional_interest_points = additional_interest_points[additional_interest_points.isin(valid_indices)]
    
            select_indices = trade_exit_bars_indices.union(interest_points_indices).union(additional_interest_points)
            select_indices = select_indices[select_indices.isin(valid_indices)].unique() # Ensure valid and unique

            equity_series_for_smoothing = equity_data_source['Equity'].copy()
            if not select_indices.empty:
                equity_series_for_smoothing = equity_series_for_smoothing.iloc[select_indices].reindex(equity_data_source.index)
                equity_series_for_smoothing.interpolate(inplace=True)
            equity = equity_series_for_smoothing # Use the smoothed series

        # Ensure equity is a pd.Series with the same RangeIndex as equity_data_source
        if not isinstance(equity, pd.Series) or not equity.index.equals(equity_data_source.index):
             equity = equity_data_source['Equity'].copy() # Fallback to non-smoothed

        # assert equity.index.equals(equity_data_source.index) # equity_data_source has RangeIndex

        if relative_equity:
            equity /= equity.iloc[0]
        if is_return:
            equity -= equity.iloc[0]

        yaxis_label = 'Return' if is_return else 'Equity'
        source_key = 'eq_return' if is_return else 'equity'
        source.add(equity, source_key)
        fig = new_indicator_figure(
            y_axis_label=yaxis_label,
            **(dict(height=80) if plot_drawdown else dict(height=100)))

        # High-watermark drawdown dents
        # Ensure equity used here is the (potentially smoothed) series on plot_index
        patch_source_df = pd.DataFrame({'index': plot_index})
        patch_source_df['equity_dd_upper'] = equity.cummax().values # equity is on plot_index
        patch_source_df['equity_dd_lower'] = equity.values

        fig.patch(x='index', y='equity_dd_lower',  # Changed y1 and y2 to y
            source=ColumnDataSource(pd.DataFrame({
                'index': np.r_[plot_index, plot_index[::-1]],
                'equity_dd_lower': np.r_[equity.values, equity.cummax().values[::-1]], # This is the y-coordinate for the patch
                 # 'equity_dd_upper': np.r_[equity.cummax().values, equity.cummax().values[::-1]] # Alternative if needed
            })),
            fill_color='#ffffea', line_color='#ffcb66', alpha=0.7) # Added alpha

        # Equity line
        r = fig.line('index', source_key, source=source, line_width=1.5, line_alpha=1, legend_label='Strategy')
        if relative_equity:
            tooltip_format = f'@{source_key}{{+0,0.[000]%}}'
            tick_format = '0,0.[00]%'
            legend_format = '{:,.0f}%'
        else:
            tooltip_format = f'@{source_key}{{$ 0,0}}'
            tick_format = '$ 0.0 a'
            legend_format = '${:,.0f}'
        set_tooltips(fig, [(yaxis_label, tooltip_format)], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

        # Buy-and-hold reference performance
        if relative_equity and not is_return:
            fig.line(
                'index',
                'bnh_perf',
                source=source,
                line_width=1,
                line_alpha=1,
                color='#666666',
                legend_label='Buy&Hold',
            )
            set_tooltips(
                fig,
                [
                    (yaxis_label, tooltip_format),
                    ('Buy&Hold', f'@bnh_perf{{+0,0.[000]%}}'),
                ],
                renderers=[r],
            )

        # Peaks
        # equity.index can be DatetimeIndex or RangeIndex. plot_index is RangeIndex.
        # df_source_data['datetime'] maps plot_index positions to original datetime values.
        
        # For peak point
        if isinstance(equity.index, pd.DatetimeIndex):
            peak_idx_val_ts = equity.idxmax() # Timestamp
            try:
                # Map Timestamp to plot_index position via df_source_data
                peak_x_coord = plot_index[df_source_data['datetime'] == peak_idx_val_ts][0]
            except IndexError: # Fallback if timestamp not found (e.g. due to resampling mismatches)
                peak_x_coord = plot_index[equity.index.get_loc(peak_idx_val_ts)]
            peak_y_coord = equity.loc[peak_idx_val_ts]
        else: # equity.index is RangeIndex
            peak_x_coord = equity.idxmax() # Position
            peak_y_coord = equity.iloc[peak_x_coord]
        
        fig.scatter(peak_x_coord, peak_y_coord,
                    legend_label='Peak ({})'.format(
                        legend_format.format(peak_y_coord * (100 if relative_equity else 1))),
                    color='cyan', size=8)

        # For final point: equity.iloc[-1] for y-value, plot_index[-1] for x-value.
        fig.scatter(plot_index[-1], equity.iloc[-1],
                    legend_label='Final ({})'.format(
                        legend_format.format(equity.iloc[-1] * (100 if relative_equity else 1))),
                    color='blue', size=8)

        if not plot_drawdown:
            drawdown_series = equity_data['DrawdownPct'] # This series' index matches equity_data's original index type
            if isinstance(drawdown_series.index, pd.DatetimeIndex):
                mdd_idx_val_ts = drawdown_series.idxmax() # Timestamp
                try:
                    mdd_x_coord = plot_index[df_source_data['datetime'] == mdd_idx_val_ts][0]
                except IndexError:
                    mdd_x_coord = plot_index[drawdown_series.index.get_loc(mdd_idx_val_ts)]
                # Y-value for scatter should be from equity at that corresponding point
                mdd_y_coord = equity.loc[mdd_idx_val_ts] if isinstance(equity.index, pd.DatetimeIndex) else equity.iloc[mdd_x_coord]
                mdd_label_val = drawdown_series.loc[mdd_idx_val_ts]
            else: # RangeIndex
                mdd_x_coord = drawdown_series.idxmax() # Position
                mdd_y_coord = equity.iloc[mdd_x_coord]
                mdd_label_val = drawdown_series.iloc[mdd_x_coord]
            
            fig.scatter(mdd_x_coord, mdd_y_coord,
                        legend_label='Max Drawdown (-{:.1f}%)'.format(100 * mdd_label_val),
                        color='red', size=8)

        # Max Drawdown Duration line
        # dd_start, dd_end are Timestamps if derived from DatetimeIndex series.
        # `equity` variable at this stage should have RangeIndex due to earlier processing.
        if dd_start is not None and dd_end is not None and not df_source_data.empty:
            try:
                # Map dd_start and dd_end (Timestamps) to plot_index positions
                datetime_index_map = pd.Index(df_source_data['datetime'])
                dd_start_pos_idx = datetime_index_map.get_indexer([dd_start], method='nearest')[0]
                dd_end_pos_idx = datetime_index_map.get_indexer([dd_end], method='nearest')[0]

                # y-value for the line start. `equity` has RangeIndex here.
                y_start_val_dd_line = equity.iloc[dd_start_pos_idx]

                start_dt_label = df_source_data['datetime'].iloc[dd_start_pos_idx]
                end_dt_label = df_source_data['datetime'].iloc[dd_end_pos_idx]
                dd_timedelta_label = end_dt_label - start_dt_label
            
                fig.line([plot_index[dd_start_pos_idx], plot_index[dd_end_pos_idx]],
                         y_start_val_dd_line,
                         line_color='red', line_width=2,
                         legend_label=f'Max Dd Dur. ({dd_timedelta_label})'.replace(' 00:00:00', '').replace('(0 days ', '('))
            except (KeyError, IndexError):
                 pass # Skip plotting max drawdown duration line if mapping fails
        
        figs_above_ohlc.append(fig)

    def _plot_equity_stack_section(relative=True):
        '''Equity stack area chart section'''
        # equity_data_source is already resampled and has RangeIndex
        equity_components = equity_data_source.iloc[:, 1:-2].copy().abs().fillna(0) # Exclude 'Equity', 'DrawdownPct', 'DrawdownDuration', 'Cash'
        equity_components = equity_components.loc[:, equity_components.sum() > 0] # Keep only components with some value
        names = list(equity_components.columns)

        if relative:
            equity_sum = equity_components.sum(axis=1)
            # Avoid division by zero if sum is zero for a row
            equity_components = equity_components.divide(equity_sum.replace(0, np.nan), axis=0).fillna(0)
        
        # equity_source_data should have 'index' column for x-axis
        equity_source_data = equity_components.copy()
        equity_source_data['index'] = plot_index # plot_index is the RangeIndex for plotting
        equity_source = ColumnDataSource(equity_source_data)


        yaxis_label = 'Allocation'
        fig = new_indicator_figure(
            y_axis_label=yaxis_label, height=max(60 + len(names), 80) # Adjusted height calculation
        )

        if relative:
            tooltip_format = [f'@{ticker}{{0,0.[000]%}}' for ticker in names] # Adjusted format
            tick_format = '0,0.[00]%'
            # For relative plot, the total stack height is 1 (or 100%)
            # Add a dummy 'total_equity_for_plot' to source if needed by tooltip logic, or adjust tooltip
            # The varea_stack implies the total.
        else:
            tooltip_format = [f'@{ticker}{{$0,0}}' for ticker in names] # Adjusted format
            tick_format = '$0.0a'
            # For absolute plot, use the actual total equity values
            # equity_source.add(equity_data_source['Equity'], 'total_equity_for_plot') # If needed for tooltips

        cg = colorgen()
        colors = [next(cg) for _ in range(len(names))]
        r = fig.line('index', 'equity', source=equity_source, line_width=1, line_alpha=0)
        fig.varea_stack(
            stackers=names,
            x='index',
            color=colors,
            legend_label=names,
            source=equity_source,
        )
        set_tooltips(fig, list(zip(names, tooltip_format)), renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

        figs_above_ohlc.append(fig)

    def _plot_drawdown_section():
        """Drawdown section"""
        fig = new_indicator_figure(y_axis_label="Drawdown", height=80)
        # equity_data_source has DrawdownPct and RangeIndex
        drawdown_series = equity_data_source['DrawdownPct']
        argmax_pos = drawdown_series.idxmax() # Positional index
        
        # Add to main plot source (source) which uses plot_index (RangeIndex)
        source.add(drawdown_series.values, 'drawdown') # Add as numpy array to match plot_index

        r = fig.line('index', 'drawdown', source=source, line_width=1.3)
        fig.scatter(plot_index[argmax_pos], drawdown_series.iloc[argmax_pos], # Use plot_index for x-coordinate
                    legend_label='Peak (-{:.1f}%)'.format(100 * drawdown_series.iloc[argmax_pos]),
                    color='red', size=8)
        set_tooltips(fig, [('Drawdown', '@drawdown{-0.0%}')], renderers=[r]) # Adjusted format
        fig.yaxis.formatter = NumeralTickFormatter(format="-0.[0]%")
        return fig

    def _plot_pl_section():
        """Profit/Loss markers section"""
        fig = new_indicator_figure(y_axis_label="Profit / Loss", height=80)
        fig.add_layout(Span(location=0, dimension='width', line_color='#666666',
                            line_dash='dashed', level='underlay', line_width=1))
        trade_source.add(trades['ReturnPct'], 'returns')
        size = trades['Size'].abs()
        size = np.interp(size, (size.min(), size.max()), (8, 20))
        trade_source.add(size, 'marker_size')
        if 'count' in trades:
            trade_source.add(trades['count'], 'count')
        trade_source.add(trades[['EntryBar', 'ExitBar']].values.tolist(), 'lines')
        fig.multi_line(xs='lines',
                       ys=transform('returns', CustomJSTransform(v_func='return [...xs].map(i => [0, i]);')),
                       source=trade_source, color='#999', line_width=1)
        r1 = fig.scatter('index', 'returns', source=trade_source, fill_color=cmap,
                         marker='circle', line_color='black', size='marker_size')
        tooltips = [("Size", "@size{0,0}")]
        if 'count' in trades:
            tooltips.append(("Count", "@count{0,0}"))
        set_tooltips(fig, tooltips + [("P/L", "@returns{+0.[000]%}")],
            vline=False, renderers=[r1])
        fig.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
        return fig

    def _plot_volume_section():
        """Volume section"""
        fig = new_indicator_figure(height=70, y_axis_label="Volume")
        fig.yaxis.ticker.desired_num_ticks = 3
        # Ensure x-axis formatter is correctly assigned if fig_ohlc.xaxis[0] is used
        if fig_ohlc.xaxis: # Check if xaxis exists
            fig.xaxis.formatter = fig_ohlc.xaxis[0].formatter
        fig.xaxis.visible = True
        if fig_ohlc.xaxis:
            fig_ohlc.xaxis.visible = False  # Show only Volume's xaxis
        
        # 'Volume' is in the main `source` (from aggregated df_source_data)
        r = fig.vbar('index', BAR_WIDTH, 'Volume', source=source, color=inc_cmap)
        set_tooltips(fig, [('Volume', '@Volume{0.00a}')], renderers=[r]) # Adjusted format
        fig.yaxis.formatter = NumeralTickFormatter(format="0 a")
        return fig

    def _plot_superimposed_ohlc():
        """Superimposed, downsampled vbars"""
    # df_source_data contains 'datetime' column with original (possibly resampled) datetime index
        if not isinstance(df_source_data['datetime'], pd.DatetimeIndex) and not pd.api.types.is_datetime64_any_dtype(df_source_data['datetime']):
            warnings.warn(
                'Superimposing requires a datetime index. Skipping.',
                stacklevel=4,
            )
            return

        datetime_series_for_superimpose = pd.to_datetime(df_source_data['datetime'])
        # Convert Series to DatetimeIndex to access 'resolution' attribute
        time_resolution_index = pd.DatetimeIndex(datetime_series_for_superimpose)
        time_resolution = time_resolution_index.resolution
        
        resample_rule_map = dict(day='ME', hour='D', minute='h', second='min', millisecond='s')
        # Ensure time_resolution is a string key in resample_rule_map
        time_resolution_key = str(time_resolution).lower() # Normalize if needed

        resample_rule_val = (superimpose if isinstance(superimpose, str) else
                             resample_rule_map.get(time_resolution_key))
        
        if not resample_rule_val:
            warnings.warn(
                f"Can't superimpose OHLC data with rule '{resample_rule_val}' "
                f"(index datetime resolution: '{time_resolution_key}'). Skipping.",
                stacklevel=4)
            return

        # Use the original df (aggregated, before reset_index) for resampling
        # df is the aggregated, possibly resampled, DataFrame with DateTimeIndex
        df_for_superimpose = df.assign(_width=1) # df still has DateTimeIndex here
        
        df2 = (df_for_superimpose.resample(resample_rule_val, label='left')
               .agg(dict(OHLCV_AGG, _width='count')))

        if df2.empty:
            warnings.warn('Superimposed OHLC data is empty after resampling. Skipping.', stacklevel=4)
            return
            
        orig_freq = _data_period(df.index) # df has DateTimeIndex
        resample_freq = _data_period(df2.index) # df2 also has DateTimeIndex
        
        if resample_freq < orig_freq: # Ensure it's a valid pd.Timedelta comparison
            if isinstance(orig_freq, Number) or isinstance(resample_freq, Number): # Handle non-Timedelta case
                 pass # Cannot compare if not Timedelta, skip check
            elif resample_freq < orig_freq:
                raise ValueError('Invalid value for `superimpose`: Upsampling not supported.')
        
        if orig_freq == resample_freq: # Ensure it's a valid pd.Timedelta comparison
            if not (isinstance(orig_freq, Number) or isinstance(resample_freq, Number)) and orig_freq == resample_freq:
                 warnings.warn('Superimposed OHLC plot matches the original plot. Skipping.', stacklevel=4)
                 return

        # Map df2's DateTimeIndex to the plot_index (RangeIndex)
        # This requires finding the corresponding plot_index positions for df2's start times
        # df_source_data['datetime'] holds the mapping from plot_index to original datetimes
        
        # Create a mapping from datetime to plot_index
        datetime_to_plot_idx = pd.Series(df_source_data.index, index=df_source_data['datetime'])
        
        # Map start of each resampled bar in df2 to a plot_index
        # This assumes df2.index are the start timestamps of the superimposed bars
        df2_plot_indices = datetime_to_plot_idx.reindex(df2.index, method='ffill').fillna(-1).astype(int)
        
        # Filter out entries that couldn't be mapped or are out of bounds
        valid_map = (df2_plot_indices >= 0) & (df2_plot_indices < len(plot_index))
        df2 = df2[valid_map]
        df2_plot_indices = df2_plot_indices[valid_map]

        if df2.empty:
            warnings.warn('Superimposed OHLC data is empty after mapping to plot index. Skipping.', stacklevel=4)
            return

        # The 'index' for source2 should be the plot_index positions
        df2['plot_idx_start'] = df2_plot_indices
        # Width in terms of plot_index units. This is tricky.
        # If _width is count of original bars, map end time to plot_index too.
        # For simplicity, let's assume _width is roughly constant in plot_index units.
        # This might need adjustment based on how sparse the original data is relative to superimposed.
        # A simple approach: width of the superimposed bar is proportional to its duration.
        # Or, use the number of original bars it spans (_width) if that's meaningful on plot_index.
        
        # Let's use the number of original candles spanned as width on the linear plot_index
        df2['plot_width'] = df2['_width'] * BAR_WIDTH # Approximate width on plot_index
        df2['plot_idx_center'] = df2['plot_idx_start'] + df2['plot_width'] / 2 - (BAR_WIDTH/2) # Centering
        df2['plot_width'] -= 0.1 # Ensure candles don't touch if BAR_WIDTH is used

        df2['inc'] = (df2.Close >= df2.Open).astype(int).astype(str)
        
        source2 = ColumnDataSource(df2)
        fig_ohlc.segment('plot_idx_center', 'High', 'plot_idx_center', 'Low', source=source2, color='#bbbbbb')
        colors_lighter = [lightness(BEAR_COLOR, .92), lightness(BULL_COLOR, .92)]
        fig_ohlc.vbar('plot_idx_center', 'plot_width', 'Open', 'Close', source=source2, line_color=None,
                      fill_color=factor_cmap('inc', colors_lighter, ['0', '1']))

    def _plot_ohlc():
        """Main OHLC bars"""
        fig_ohlc.segment('index', 'High', 'index', 'Low', source=source, color="black",
                         legend_label='OHLC')
        r = fig_ohlc.vbar('index', BAR_WIDTH, 'Open', 'Close', source=source,
                          line_color="black", fill_color=inc_cmap, legend_label='OHLC')
        return r

    def _plot_ohlc_trades():
        """Trade entry / exit markers on OHLC plot"""
        if not trades.empty:
            trade_source.add(trades[['EntryBar', 'ExitBar']].values.tolist(), 'position_lines_xs')
            trade_source.add( trades[['EntryPrice', 'ExitPrice']].values.tolist(), 'position_lines_ys')
        else:
            trade_source.add([], 'position_lines_xs')
            trade_source.add([], 'position_lines_ys')

        fig_ohlc.multi_line(xs='position_lines_xs', ys='position_lines_ys',
                            source=trade_source, line_color=trades_cmap,
                            legend_label=f'Trades ({len(trades)})',
                            line_width=8, line_alpha=1, line_dash='dotted')

    def _plot_ohlc_universe():
        # data is Dict[str, pd.DataFrame], df_source_data is from aggregated df
        # We plot individual ticker 'Close' prices from the data_dict
        # These need to be aligned with plot_index (RangeIndex)
        fig = fig_ohlc
        ohlc_colors = colorgen()
        label_tooltip_pairs = []
        
        tickers_to_plot = list(data.keys())[:10]
        num_total_tickers = len(data.keys())

        for ticker in tickers_to_plot:
            color = next(ohlc_colors)
            source_name = f"{ticker}_Close" # Unique source name
            
            # Get the 'Close' series for the ticker, ensure it's aligned with plot_index
            # data[ticker] is the original (possibly resampled) DataFrame for the ticker
            # It should have the same DateTimeIndex as the aggregated `df` before `df` was reset.
            # So, we reindex it to `df.index` (which is the DateTimeIndex of aggregated, resampled data)
            # then reset_index to get values aligned with `plot_index`.
            if ticker in data and 'Close' in data[ticker].columns:
                ticker_close_series = data[ticker]['Close'].reindex(df.index).reset_index(drop=True)
                
                source.add(ticker_close_series.values, source_name) # Add as numpy array
                label_tooltip_pairs.append((ticker, f'@{{{source_name}}}{{0,0.0[0000]}}'))
                
                # ohlc_extreme_values is used for y-axis auto-scaling of the main OHLC plot
                # Add this ticker's close to it.
                ohlc_extreme_values[source_name] = ticker_close_series.values

                fig.line(
                    'index', # x-coordinates are from plot_index (0 to N-1)
                    source_name,
                    source=source,
                    legend_label=ticker, # Legend uses actual ticker name
                    line_color=color,
                    line_width=2,
                )
        
        ohlc_tooltips.extend(label_tooltip_pairs)
        if num_total_tickers > 10:
            fig.line(
                x=[0], y=[0], # Dummy line for legend entry
                legend_label=f'{num_total_tickers - 10} more tickers hidden',
                line_color='black', visible=False # Make it invisible but keep legend
            )
        
        fig.legend.orientation = 'horizontal'
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_alpha = 0

    def _plot_indicators():
        """Strategy indicators"""

        def _too_many_dims(value):
            assert value.ndim >= 2
            if value.ndim > 2:
                warnings.warn(f"Can't plot indicators with >2D ('{value.name}')",
                              stacklevel=5)
                return True
            return False

        class LegendStr(str):
            # The legend string is such a string that only matches
            # itself if it's the exact same object. This ensures
            # legend items are listed separately even when they have the
            # same string contents. Otherwise, Bokeh would always consider
            # equal strings as one and the same legend item.
            def __eq__(self, other):
                return self is other

        ohlc_colors = colorgen()
        indicator_figs = []

        for i, value in enumerate(indicators):
            value = np.atleast_2d(value)

            # Use .get()! A user might have assigned a Strategy.data-evolved
            # _Array without Strategy.I()
            if not value._opts.get('plot') or _too_many_dims(value):
                continue

            is_overlay = value._opts['overlay']
            is_scatter = value._opts['scatter']
            if is_overlay:
                fig = fig_ohlc
            else:
                fig = new_indicator_figure()
                indicator_figs.append(fig)
            tooltips = []
            colors = value._opts['color']
            colors = colors and cycle(_as_list(colors)) or (
                cycle([next(ohlc_colors)]) if is_overlay else colorgen())

            if isinstance(value.name, str):
                tooltip_label = value.name
                legend_labels = [LegendStr(value.name)] * len(value)
            else:
                tooltip_label = ", ".join(value.name)
                legend_labels = [LegendStr(item) for item in value.name]

            for j, arr in enumerate(value):
                color = next(colors)
                source_name = f'{legend_labels[j]}_{i}_{j}'
                if arr.dtype == bool:
                    arr = arr.astype(int)
                source.add(arr, source_name)
                tooltips.append(f'@{{{source_name}}}{{0,0.0[0000]}}')
                if is_overlay:
                    ohlc_extreme_values[source_name] = arr
                    if is_scatter:
                        fig.circle(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], color=color,
                            line_color='black', fill_alpha=.8,
                            radius=BAR_WIDTH / 2 * .9)
                    else:
                        fig.line(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], line_color=color,
                            line_width=1.3)
                else:
                    if is_scatter:
                        r = fig.circle(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], color=color,
                            radius=BAR_WIDTH / 2 * .6)
                    else:
                        r = fig.line(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], line_color=color,
                            line_width=1.3)
                    # Add dashed centerline just because
                    mean = try_(lambda: float(pd.Series(arr).mean()), default=np.nan)
                    if not np.isnan(mean) and (abs(mean) < .1 or
                                               round(abs(mean), 1) == .5 or
                                               round(abs(mean), -1) in (50, 100, 200)):
                        fig.add_layout(Span(location=float(mean), dimension='width',
                                            line_color='#666666', line_dash='dashed',
                                            level='underlay', line_width=.5))
            if is_overlay:
                ohlc_tooltips.append((tooltip_label, NBSP.join(tooltips)))
            else:
                set_tooltips(fig, [(tooltip_label, NBSP.join(tooltips))], vline=True, renderers=[r])
                # If the sole indicator line on this figure,
                # have the legend only contain text without the glyph
                if len(value) == 1:
                    fig.legend.glyph_width = 0
        return indicator_figs

    # Construct figure ...

    if plot_equity:
        _plot_equity_section()

    if plot_allocation:
        _plot_equity_stack_section(relative_allocation)

    if plot_return:
        _plot_equity_section(is_return=True)

    if plot_drawdown:
        figs_above_ohlc.append(_plot_drawdown_section())

    if plot_pl:
        figs_above_ohlc.append(_plot_pl_section())

    if plot_volume:
        fig_volume = _plot_volume_section()
        figs_below_ohlc.append(fig_volume)

    if superimpose and is_datetime_index:
        _plot_superimposed_ohlc()

    ohlc_bars = _plot_ohlc()
    # data is Dict[str, pd.DataFrame]
    if plot_trades and len(data.keys()) <= 10: # data.keys() gives tickers
        _plot_ohlc_trades()
    if len(data.keys()) > 1:
        _plot_ohlc_universe()
    if plot_indicator:
        indicator_figs = _plot_indicators()
        if reverse_indicators:
            indicator_figs = indicator_figs[::-1]
        figs_below_ohlc.extend(indicator_figs)

    set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[ohlc_bars])

    ohlc_low_col = ohlc_extreme_values.min(1, skipna=True) if not ohlc_extreme_values.empty else pd.Series(dtype=float)
    ohlc_high_col = ohlc_extreme_values.max(1, skipna=True) if not ohlc_extreme_values.empty else pd.Series(dtype=float)
    source.add(ohlc_low_col, 'ohlc_low')
    source.add(ohlc_high_col, 'ohlc_high')


    custom_js_args = dict(ohlc_range=fig_ohlc.y_range,
                          source=source)
    if plot_volume:
        if 'fig_volume' in locals():
            custom_js_args.update(volume_range=fig_volume.y_range)

    fig_ohlc.x_range.js_on_change('end', CustomJS(args=custom_js_args,
                                                  code=_AUTOSCALE_JS_CALLBACK))

    figs = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc
    linked_crosshair = CrosshairTool(
        dimensions='both', line_color='lightgrey',
        overlay=(Span(dimension="width", line_dash="dotted", line_width=1),
                 Span(dimension="height", line_dash="dotted", line_width=1)),
    )

    for f in figs:
        if f.legend:
            f.legend.visible = show_legend
            f.legend.location = 'top_left'
            f.legend.border_line_width = 1
            f.legend.border_line_color = '#333333'
            f.legend.padding = 5
            f.legend.spacing = 0
            f.legend.margin = 0
            f.legend.label_text_font_size = '8pt'
            f.legend.click_policy = "hide"
            f.legend.background_fill_alpha = .9
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'

        f.add_tools(linked_crosshair)
        wheelzoom_tool = next(wz for wz in f.tools if isinstance(wz, WheelZoomTool))
        wheelzoom_tool.maintain_focus = False

    kwargs_grid = {}
    if plot_width is None:
        kwargs_grid['sizing_mode'] = 'stretch_width'

    fig = gridplot(
        figs,
        ncols=1,
        toolbar_location='right',
        toolbar_options=dict(logo=None),
        merge_tools=True,
        **kwargs_grid,  # type: ignore
    )
    show(fig, browser=None if open_browser else 'none')
    return fig


def plot_heatmaps(heatmap: pd.Series, agg: Union[Callable, str], ncols: int,
                  filename: str = '', plot_width: int = 1200, open_browser: bool = True):
    if not (isinstance(heatmap, pd.Series) and
            isinstance(heatmap.index, pd.MultiIndex)):
        raise ValueError('heatmap must be heatmap Series as returned by '
                         '`Backtest.optimize(..., return_heatmap=True)`')

    _bokeh_reset(_windos_safe_filename(filename) if filename else None)

    param_combinations = combinations(heatmap.index.names, 2)
    dfs = [heatmap.groupby(list(dims)).agg(agg).to_frame(name='_Value')
        for dims in param_combinations]
    if not dfs or all(df.empty for df in dfs):
         warnings.warn('No data to plot in heatmap.')
         return None

    figs = []
    valid_dfs = [df for df in dfs if not df.empty and not df['_Value'].isnull().all()]
    if not valid_dfs:
         warnings.warn('All heatmap data is NaN. Cannot determine color range.')
         cmap_low, cmap_high = 0, 1
    else:
         cmap_low = min(df['_Value'].min() for df in valid_dfs)
         cmap_high = max(df['_Value'].max() for df in valid_dfs)
         if cmap_low == cmap_high:
             cmap_low -= 0.1
             cmap_high += 0.1
             if cmap_low == cmap_high:
                 cmap_high += 1e-6

    cmap = LinearColorMapper(palette='Viridis256',
                             low=cmap_low, high=cmap_high,
                             nan_color='white',
    )
    for df in dfs:
        if df.empty:
            continue
        name1, name2 = df.index.names
        level1 = df.index.levels[0].astype(str).tolist() if len(df.index.levels) > 0 else []
        level2 = df.index.levels[1].astype(str).tolist() if len(df.index.levels) > 1 else []
        df = df.reset_index()
        df[name1] = df[name1].astype('str')
        df[name2] = df[name2].astype('str')

        fig = _figure(x_range=level1,  # type: ignore[call-arg]
                      y_range=level2,
                      x_axis_label=name1,
                      y_axis_label=name2,
                      width=plot_width // ncols,
                      height=plot_width // ncols,
                      tools='box_zoom,reset,save',
                      tooltips=[(name1, '@' + name1),
                                (name2, '@' + name2),
                                ('Value', '@_Value{0.[000]}')])
        fig.grid.grid_line_color = None        # type: ignore[attr-defined]
        fig.axis.axis_line_color = None        # type: ignore[attr-defined]
        fig.axis.major_tick_line_color = None  # type: ignore[attr-defined]
        fig.axis.major_label_standoff = 0      # type: ignore[attr-defined]

        fig.rect(
            x=name1,
            y=name2,
            width=1,
            height=1,
            source=df,
            line_color=None,
            fill_color=dict(field='_Value',
                            transform=cmap))
        fig.toolbar.logo = None
        figs.append(fig)

    if not figs:
        warnings.warn('No valid heatmaps generated.')
        return None

    fig = gridplot(
        figs,  # type: ignore
        ncols=ncols,
        toolbar_location='above',
        merge_tools=True,
    )
    show(fig, browser=None if open_browser else 'none')
    return fig
