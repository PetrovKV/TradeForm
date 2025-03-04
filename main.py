from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
import pandas as pd
import os
import alpaca_trade_api as tradeapi
from pytz import timezone, UTC
import time  # Для измерения времени выполнения
from dotenv import load_dotenv

# Инициализация приложения
app = FastAPI()
#root_path="/TradeForm"

# Загрузка переменных окружения
load_dotenv()

# Настройка папок
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

templates = Jinja2Templates(directory="templates")

# Настройка Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


@app.get("/TradeForm", response_class=HTMLResponse)
async def get_data_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/TradeForm/fetch-data/", response_class=HTMLResponse)
async def fetch_data(
    request: Request,
    ticker: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...)
):
    try:
        total_start_time = time.time() #замер времени
        # Получение данных с Alpaca
        api_start_time = time.time() #замер времени
        barset = api.get_bars(
            ticker,
            timeframe="5Min",
            start=start_date,
            end=end_date,
            adjustment = 'split'
        ).df
        api_duration = time.time() - api_start_time #замер времени
        print(f"Время выполнения запроса данных с API: {api_duration:.2f} секунд")

        if barset.empty:
            raise ValueError("Не удалось получить данные для указанного тикера.")

        processing_start_time = time.time()#замер времени

        # Преобразование данных в DataFrame
        barset = barset.reset_index()
        barset["timestamp"] = barset["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        barset["date"] = pd.to_datetime(barset["timestamp"]).dt.date

        # Добавление времени первой свечи для каждого дня
        first_candle_times = barset.groupby("date").first().reset_index()
        first_candle_times["FCTime"] = pd.to_datetime(first_candle_times["timestamp"]).dt.strftime("%H:%M")

        # Расчёт времени UTC, соответствующего 9:30 утра по Нью-Йорку
        ny_zone = timezone("America/New_York")
        utc_zone = UTC
        barset["time"] = pd.to_datetime(barset["timestamp"]).dt.time
        first_candle_times = barset.groupby("date").first().reset_index()
        first_candle_times["FCTime"] = first_candle_times["date"].apply(
            lambda d: ny_zone.localize(datetime.combine(d, datetime.min.time()) + timedelta(hours=9, minutes=30))
            .astimezone(utc_zone)
            .strftime("%H:%M")
        )

        # Добавление цен и вычисление новых столбцов
        first_candle_times["FCOpen"] = first_candle_times.apply(
            lambda row: round(barset[(barset["date"] == row["date"]) &
                                     (barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time())]["open"].iloc[
                                  0], 2)
            if not barset[(barset["date"] == row["date"]) &
                          (barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time())].empty else None, axis=1
        )

        # Исключение дней, где FCOpen = NaN (выходные или нерабочие дни)
        first_candle_times = first_candle_times[first_candle_times["FCOpen"].notna()]


        first_candle_times["FCClose"] = first_candle_times.apply(
            lambda row: round(barset[(barset["date"] == row["date"]) &
                                     (barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time())][
                                  "close"].iloc[0], 2)
            if not barset[(barset["date"] == row["date"]) &
                          (barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time())].empty else None, axis=1
        )
        first_candle_times["dCO"] = first_candle_times["FCClose"] - first_candle_times["FCOpen"]
        first_candle_times["dCO"] = first_candle_times["dCO"].round(2)
        first_candle_times["dCOmin"] = 0.05

        first_candle_times["TradeType"] = first_candle_times.apply(
            lambda row: (
                "LongTrade" if row["dCO"] >= row["dCOmin"] else
                "ShortTrade" if row["dCO"] <= -row["dCOmin"] else
                "NoTrade"
            ),
            axis=1
        )

        def calculate_stop_loss_and_profit_take(row, barset):
            if row["TradeType"] == "LongTrade":
                stop_loss = barset.loc[
                    (barset["date"] == row["date"]) & (
                                barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time()),
                    "low"
                ].iloc[0] if not barset.empty else None
                profit_take = row["FCClose"] + 10 * (row["FCClose"] - stop_loss)
            elif row["TradeType"] == "ShortTrade":
                stop_loss = barset.loc[
                    (barset["date"] == row["date"]) & (
                                barset["time"] == datetime.strptime(row["FCTime"], "%H:%M").time()),
                    "high"
                ].iloc[0] if not barset.empty else None
                profit_take = row["FCClose"] + 10 * (row["FCClose"] - stop_loss)
            else:
                stop_loss, profit_take = "-", "-"
            return stop_loss, profit_take

        # Добавляем столбцы Low и High для обработки в first_candle_times
        first_candle_times["Low"] = first_candle_times["date"].apply(
            lambda d: barset[(barset["date"] == d)]["low"].min()
        )
        first_candle_times["High"] = first_candle_times["date"].apply(
            lambda d: barset[(barset["date"] == d)]["high"].max()
        )

        # Обновляем StopLoss и ProfitTake
        first_candle_times[["StopLoss", "ProfitTake"]] = first_candle_times.apply(
            lambda row: pd.Series(calculate_stop_loss_and_profit_take(row, barset)), axis=1
        )

        # Округляем ProfitTake до двух знаков после запятой
        first_candle_times["ProfitTake"] = first_candle_times["ProfitTake"].apply(
            lambda x: round(x, 2) if isinstance(x, (int, float)) else x
        )

        # Округляем StopLoss до двух знаков после запятой
        first_candle_times["StopLoss"] = first_candle_times["StopLoss"].apply(
            lambda x: round(x, 2) if isinstance(x, (int, float)) else x
        )

        # Функция для определения CPTime и CPPrice
        def calculate_cp_time_and_price(row, barset):
            if row["TradeType"] == "NoTrade":
                return "-", "-"

            # Параметры для анализа
            fctime = datetime.strptime(row["FCTime"], "%H:%M").time()
            trade_date = row["date"]
            extended_time_limit = (datetime.combine(trade_date, fctime) + timedelta(hours=6, minutes=25)).time()

            # Фильтруем свечи текущего дня после FCTime до времени лимита
            daily_bars = barset[(barset["date"] == trade_date) & (barset["time"] > fctime)]
            daily_bars = daily_bars[daily_bars["time"] <= extended_time_limit]

            # Логика для LongTrade
            if row["TradeType"] == "LongTrade":
                for _, candle in daily_bars.iterrows():
                    if candle["high"] >= row["ProfitTake"]:
                        return candle["time"], row["ProfitTake"]
                    if candle["low"] <= row["StopLoss"]:
                        return candle["time"], row["StopLoss"]

            # Логика для ShortTrade
            if row["TradeType"] == "ShortTrade":
                for _, candle in daily_bars.iterrows():
                    if candle["low"] <= row["ProfitTake"]:
                        return candle["time"], row["ProfitTake"]
                    if candle["high"] >= row["StopLoss"]:
                        return candle["time"], row["StopLoss"]

            # Если ProfitTake и StopLoss не сработали
            extended_bar = barset[
                (barset["date"] == trade_date) & (barset["time"] == extended_time_limit)
                ]
            if not extended_bar.empty:
                return extended_time_limit, extended_bar["open"].iloc[0]

            return "-", "-"

        # Добавляем столбцы CPTime и CPPrice
        first_candle_times[["CPTime", "CPPrice"]] = first_candle_times.apply(
            lambda row: pd.Series(calculate_cp_time_and_price(row, barset)), axis=1
        )

        # Функция для расчёта TradeResult
        def calculate_trade_result(row):
            if row["TradeType"] == "NoTrade":
                return "-"
            if row["CPPrice"] == row["ProfitTake"]:
                return "ProfitTake"
            if row["CPPrice"] == row["StopLoss"]:
                return "StopLoss"
            return "EndDay"

        # Добавляем столбец TradeResult
        first_candle_times["TradeResult"] = first_candle_times.apply(calculate_trade_result, axis=1)

        # Функция для расчёта Income
        def calculate_income(row):
            if row["TradeType"] == "LongTrade":
                return row["CPPrice"] - row["FCClose"]
            elif row["TradeType"] == "ShortTrade":
                return row["FCClose"] - row["CPPrice"]
            return 0  # Для NoTrade возвращаем 0

        # Функция для расчёта Income%
        def calculate_income_percentage(row):
            try:
                if row["FCClose"]:  # Проверяет, что FCClose существует и не равен 0
                    return (row["Income"] / row["FCClose"]) * 100
            except KeyError:
                pass  # Если ключ отсутствует, возвращаем 0
            return 0  # Для NoTrade или в случае ошибки возвращаем 0

        # Добавляем столбцы Income и IncomePr
        first_candle_times["Income"] = first_candle_times.apply(calculate_income, axis=1)
        first_candle_times["IncomePr"] = first_candle_times.apply(calculate_income_percentage, axis=1)

        # Округляем Income и IncomePr до двух знаков после запятой
        first_candle_times["Income"] = first_candle_times["Income"].apply(
            lambda x: round(x, 2) if isinstance(x, (int, float)) else x
        )
        first_candle_times["IncomePr"] = first_candle_times["IncomePr"].apply(
            lambda x: round(x, 2) if isinstance(x, (int, float)) else x
        )

        # Применение форматирования на основе значения IncomePr
        def format_income_percentage(value):
            if pd.isna(value) or value == 0:
                return ""  # Оставляем стандартное форматирование
            elif value > 0:
                return 'background-color: green; color: white;'
            elif value < 0:
                return 'background-color: red; color: white;'

        # Применение стилей к DataFrame
        styled_df = first_candle_times.style.map(format_income_percentage, subset=["IncomePr"])

        # Передача обновлённых данных в шаблон
        fctime_data = first_candle_times[[
            "date", "FCTime", "FCOpen", "FCClose", "dCO", "dCOmin", "TradeType", "StopLoss", "ProfitTake","CPTime", "CPPrice", "TradeResult", "Income", "IncomePr"
        ]].to_dict(orient="records")


        # Сохранение в CSV
        csv_filename = f"{DATA_DIR}/{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        barset.to_csv(csv_filename, index=False)

        unique_dates = sorted(barset["date"].unique())

        # Расчёт времени завершения обработки
        processing_duration = time.time() - processing_start_time
        print(f"Время обработки данных: {processing_duration:.2f} секунд")

        # Итоговое время выполнения
        total_duration = time.time() - total_start_time
        print(f"Общее время выполнения запроса: {total_duration:.2f} секунд")

        return templates.TemplateResponse("form.html", {
            "request": request,
            "ticker": ticker,
            "data": barset.to_dict(orient="records"),
            "unique_dates": unique_dates,
            "fctime_data": fctime_data,
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": str(e),
        })


@app.get("/TradeForm/chart-data/{ticker}/{date}", response_class=JSONResponse)
async def get_chart_data(ticker: str, date: str, include_charts: bool = True):
    try:
        chart_start_time = time.time()  # замер времени
        files = [f for f in os.listdir(DATA_DIR) if ticker in f]
        if not files:
            return JSONResponse(content={"error": "Нет данных для указанного тикера."}, status_code=404)

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))
        file_path = os.path.join(DATA_DIR, latest_file)

        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))  # Формат даты

        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
        filtered_data = df[df["date"] == selected_date.strftime("%Y-%m-%d")]

        # время выполнения чарта
        total_chart_duration = time.time() - chart_start_time
        print(f"Общее время на график: {total_chart_duration:.2f} секунд")

        if filtered_data.empty:
            return JSONResponse(content={"error": "Данные для указанной даты отсутствуют."}, status_code=404)

        # Если графики не нужны, вернуть только базовые данные
        if not include_charts:
            return JSONResponse(content={"data_available": True})

        return JSONResponse(content={
            "timestamp": filtered_data["timestamp"].tolist(),
            "open": filtered_data["open"].tolist(),
            "high": filtered_data["high"].tolist(),
            "low": filtered_data["low"].tolist(),
            "close": filtered_data["close"].tolist(),
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
