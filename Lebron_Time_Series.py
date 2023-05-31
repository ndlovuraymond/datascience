import pandas as pd
from sklearn import metrics
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from dash import Dash, dcc, html, callback, Output, Input
import numpy as np

lebron_stats = pd.read_csv("lebron_career.csv",parse_dates=["date"])
lebron_stats["Year"] = lebron_stats.date.dt.year
lebron_stats["Month"] = lebron_stats.date.dt.month

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Lebron Statistics'),
    dcc.Dropdown(lebron_stats.Year.unique(),2011,id="dropdown"),
    dcc.Graph(
        id="graph"
    ),
    html.H4(id="analysis-of-forecast-MAE"),
    html.H4(id="analysis-of-forecast-MSE"),
    html.H4(id="analysis-of-forecast-RMSE")
])

@callback(
    Output('graph', 'figure'),
    Output('analysis-of-forecast-MAE', 'children'),
    Output('analysis-of-forecast-MSE', 'children'),
    Output('analysis-of-forecast-RMSE', 'children'),
    Input('dropdown', 'value')
)
def update_graph(value):
    train_data = lebron_stats.query(f"Year == {value} and Month < 6").iloc[:-5]
    test_data = lebron_stats.query(f"Year == {value} and Month < 6").iloc[-5:]
    # Fit the ARIMA model
    model = ARIMA(train_data["pts"], order=(15, 1, 1))  #Order AR is comparing previous 10 weeks of scoring
    model_fit = model.fit()

    # Forecast 5 years ahead
    forecast = model_fit.forecast(steps=5)  # Forecasting 4 weeks
    df = forecast.to_frame(name="pts")
    # Creating yearly forecast for pts for the next year
    forecast_dates = pd.Series(test_data.date)
    forecast_dates =forecast_dates.to_frame()
    forecast_dates.reset_index(inplace=True)
    forecast_pts = pd.Series(forecast.array,index=range(0,5))
    forecast_df = pd.concat([forecast_pts.rename("pts"),forecast_dates["date"]],axis=1)
    forecast_df.set_index("date",inplace=True)
    fig = px.line(lebron_stats.query(f"Year == {value} and Month < 6"),x="date",y="pts",
        labels={"pts":"Points Per Game"},title="Points Per Game").update_layout(
        paper_bgcolor="white",plot_bgcolor="white",title={"x":.5,"y":.85,"font":{"size":30}}
        ).update_yaxes(
        gridcolor="black")
    fig.add_scatter(x=forecast_df.index,y=forecast_df["pts"],mode="lines")
    MAE = "MAE: "+str(metrics.mean_absolute_error(test_data["pts"],forecast_df["pts"]))
    MSE = "MSE: "+str(metrics.mean_squared_error(test_data["pts"],forecast_df["pts"]))
    RMSE = "RMSE: "+str(np.sqrt(metrics.mean_squared_error(test_data["pts"],forecast_df["pts"])))
    return fig,MAE, MSE, RMSE


if __name__ == '__main__':
    app.run_server(debug=True)
