import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# 1. '가상의 기상청 데이터' 생성
# ---------------------------------------------------------
def generate_weather_data():
    np.random.seed(42)
    days = np.arange(1, 366)
    
    # 기온: 겨울(-10도) -> 여름(35도) -> 겨울(-10도)
    temperature = -10 + 22.5 * (1 - np.cos((days - 15) * 2 * np.pi / 365)) 
    temperature += np.random.normal(0, 2, 365)

    # 습도
    humidity = 40 + 30 * (1 - np.cos((days - 15) * 2 * np.pi / 365))
    humidity += np.random.normal(0, 5, 365)

    # 냉난방 부하 (V-Curve)
    power_usage = []
    for t, h in zip(temperature, humidity):
        base_load = 300
        
        if t < 18: # 난방 구간
            heating = (18 - t) * 12
            load = base_load + heating
        elif t > 24: # 냉방 구간
            cooling = (t - 24) * 15
            load = base_load + cooling + (h * 0.5) 
        else: # 쾌적 구간
            load = base_load
            
        load += np.random.randint(-20, 20)
        power_usage.append(load)

    return pd.DataFrame({
        '날짜': days,
        '기온(°C)': temperature,
        '습도(%)': humidity,
        '전력소비량(kWh)': power_usage
    })

# 데이터 로드 및 학습
df = generate_weather_data()
X = df[['기온(°C)', '습도(%)']]
y = df['전력소비량(kWh)']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------
# 2. Streamlit UI (수정된 부분)
# ---------------------------------------------------------
st.set_page_config(layout="wide")

# 요청하신 제목 수정
st.title("⚡ 과거 데이터 기반 전력 수요 시뮬레이션")

# 요청하신 부제목 수정 및 글자 크기 축소 (### -> ####)
st.markdown("#### 📅 2024년 기상청 데이터 기반 AI 전력 수요 예측")

st.info("💡 **Insight:** 전력 소비는 춥거나 더울 때 급증하는 **'V자형 패턴'**을 보입니다.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("환경 시뮬레이션")
    st.write("내일의 예상 날씨를 설정하세요.")
    
    user_temp = st.slider("🌡️ 기온 (°C)", -20, 40, 22)
    user_humid = st.slider("💧 습도 (%)", 20, 100, 50)
    
    st.divider()
    
    pred = model.predict([[user_temp, user_humid]])[0]
    
    st.subheader("AI 예측 결과")
    st.metric(label="예상 전력 소비량", value=f"{pred:.1f} kWh", delta_color="inverse")

    if user_temp < 10:
        st.error("🔥 [난방 급증] 겨울철 전력 피크가 예상됩니다.")
    elif user_temp > 30:
        st.error("❄️ [냉방 급증] 여름철 전력 피크가 예상됩니다.")
    elif 18 <= user_temp <= 24:
        st.success("✅ [쾌적 구간] 냉난방 수요가 가장 적습니다.")
    else:
        st.warning("⚠️ 전력 사용량이 증가하고 있습니다.")

with col2:
    st.subheader("📊 데이터 분석: 기온과 전력의 상관관계")
    
    tab1, tab2 = st.tabs(["📉 V-Curve 분석", "📅 연간 패턴"])
    
    with tab1:
        st.caption("기온(X축)에 따른 전력소비(Y축) 분포 - 뚜렷한 V자 곡선을 확인하세요.")
        st.scatter_chart(df, x='기온(°C)', y='전력소비량(kWh)', color='#FF5733')
        
    with tab2:
        st.caption("1월(겨울)과 8월(여름)에 전력 사용이 높은 것을 볼 수 있습니다.")
        st.line_chart(df.set_index('날짜')[['기온(°C)', '전력소비량(kWh)']])