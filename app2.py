import streamlit as st
import pandas as pd
import plotly.express as px
import plost
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Police Dashboard ')

st.sidebar.subheader('Data')
data_context = st.sidebar.selectbox('Choose any one below', ('Victim-data-analysis', '')) 

st.sidebar.subheader('District')
district = st.sidebar.selectbox('Choose District', ('Bengaluru', 'Tumakuru','Hassan','Belagavi','Shivamoggat','Mandya')) 

# st.sidebar.subheader('Donut chart parameter')
# donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

# st.sidebar.subheader('Line chart parameters')
# plot_data = st.sidebar.multiselect('Select data', ['min', 'max'], ['min', 'max'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Victim Registered District Count", "41", "1")
col2.metric("Types of Crime Count", "11", "3")
col3.metric("Types of Victim Profession count", "161", "23")

seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')
df=pd.read_csv("bengaluru.csv")
if(data_context=="Victim-data-analysis" and district=="Bengaluru"):
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.title('Bengaluru Map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, width=500, height=400)
        folium.Marker(location=[12.9716, 77.5946], popup='Bengaluru').add_to(m)
        folium_static(m)

    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)

if(data_context=="Victim-data-analysis" and district=="Tumakuru"):
    df_tumakuru=pd.read_csv("tumakuru.csv")
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df_tumakuru['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.title('Bengaluru Map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, width=500, height=400)
        folium.Marker(location=[12.9716, 77.5946], popup='Bengaluru').add_to(m)
        folium_static(m)

    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df_tumakuru.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df_tumakuru.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df_tumakuru['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df_tumakuru.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df_tumakuru['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df_tumakuru.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)


if(data_context=="Victim-data-analysis" and district=="Hassan"):
    df_hassan=pd.read_csv("hassan.csv")
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df_hassan['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.title('Bengaluru Map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, width=500, height=400)
        folium.Marker(location=[12.9716, 77.5946], popup='Bengaluru').add_to(m)
        folium_static(m)

    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df_hassan.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df_hassan.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df_hassan['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df_hassan.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df_hassan['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df_hassan.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)


if(data_context=="Victim-data-analysis" and district=="Belagavi"):
    df_Belagavi=pd.read_csv("Belagavi.csv")
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df_Belagavi['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.title('Bengaluru Map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, width=500, height=400)
        folium.Marker(location=[12.9716, 77.5946], popup='Bengaluru').add_to(m)
        folium_static(m)

    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df_Belagavi.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df_Belagavi.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df_Belagavi['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df_Belagavi.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df_Belagavi['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df_Belagavi.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)


if(data_context=="Victim-data-analysis" and district=="Shivamoggat"):
    df_Shivamoggat=pd.read_csv("Belagavi.csv")
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df_Shivamoggat['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.title('Bengaluru Map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, width=500, height=400)
        folium.Marker(location=[12.9716, 77.5946], popup='Bengaluru').add_to(m)
        folium_static(m)

    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df_Shivamoggat.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df_Shivamoggat.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df_Shivamoggat['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df_Shivamoggat.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df_Shivamoggat['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df_Shivamoggat.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)


if(data_context=="Victim-data-analysis" and district=="Mandya"):
    df_Mandya=pd.read_csv("Mandya_df.csv")
    ps=pd.read_csv("mandya_ps.csv")
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Bar chart with counts for Unit Station')

        unit_station_counts = df_Mandya['UnitName'].value_counts().reset_index()
        unit_station_counts.columns = ['UnitName', 'Count']

        fig = px.bar(unit_station_counts, x='UnitName', y='Count', color='UnitName')

        fig.update_layout(
            title='Count of Unit Stations',
            xaxis_title='Unit Station',
            yaxis_title='Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:


        # Create a map centered around Bengaluru with satellite imagery
        st.markdown('### Mandya District crime hotspot map')
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google')

        # Add markers for each latitude and longitude in the DataFrame
        for index, row in ps.iterrows():
            folium.Marker(location=[row['Latitude'], row['Longitude']], popup=row['UnitName']).add_to(m)
        folium_static(m)


    c3,c4=st.columns((7,3))
    with c3:
        st.markdown('### Polar Chart: Average Age of Victims by Person Type')

        age_by_person_type = df_Mandya.groupby('PersonType')[['age']].mean().reset_index()

        fig_polar = px.line_polar(age_by_person_type, r='age', theta='PersonType', line_close=True, title='Average Age of Victims by Person Type')
 
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    title='Average Age'
                )
            )
        )

        st.plotly_chart(fig_polar, use_container_width=True)
    df_person_type = df_Mandya.groupby('UnitName')['PersonType'].apply(lambda x: x.mode()).reset_index()
    df_mode=df_person_type.drop('level_1',axis=1)

    with c4:
        st.markdown('### Mode of PersonType for Each UnitName')

        st.write(df_mode)

    c5,c6=st.columns((7,3))
    with c5:
        st.markdown('### Sex Distribution over Years - Pie Chart')
        sex_counts = df_Mandya['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig_pie = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution over Years')
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        df = df.dropna(subset=['Year'])
        crime_counts_by_year = df_Mandya.groupby('Year').size().reset_index(name='CrimeCount')
        fig = px.line(crime_counts_by_year, x='Year', y='CrimeCount', title='Crime Rates Over Years')
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Crimes')
        st.plotly_chart(fig, use_container_width=True)
        # st.write("Crime Statistics by Year:")
        # st.write(crime_counts_by_year)

    c7,c8=st.columns((7,3))
    with c7:
        profession_crime_counts = df_Mandya['Profession'].value_counts().reset_index(name='CrimeCount')
        fig_profession_crimes = px.bar(profession_crime_counts, x='index', y='CrimeCount', title='Profession-wise Crime Rates')
        st.plotly_chart(fig_profession_crimes)
    with c8:
        injury_type_by_sex = df_Mandya.groupby(['Sex', 'InjuryType']).size().reset_index(name='Count')
        fig_injury_type_sex = px.bar(injury_type_by_sex, x='InjuryType', y='Count', color='Sex', title='Injury Type Distribution by Sex')
        st.plotly_chart(fig_injury_type_sex)

    