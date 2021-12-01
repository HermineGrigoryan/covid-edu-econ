import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm


X_vars = ['gdp_growth_rate', 'gdp_per_cap_PPP',
       'personal_income_tax_rate', 'debt_to_gdp', 'ease_of_doing_business',
       'unemployment_rate', 'consumer_price_index',
       'inflation_rates', 'corruption_rank', 'population', 
       'gov_exp_on_edu_percent_of_gdp', 'gov_exp_on_edu_perc_per_student',
        'gov_exp_per_stud_tertiary_perc', 'pupil_teacher_ratio_primary',
        'youth_literacy_rate',
       'Continent', 'Sub Region']
X_vars = [i.replace('_', ' ').title() for i in X_vars]

Y_vars = ['covid_death_rate', 'covid_cases_rate', 'covid_vaccination_rate']
Y_vars = [i.replace('_', ' ').title() for i in Y_vars]

grouping_vars = [None, 'Continent', 'Sub Region']

# Cached functions
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1500}px
    }}
    .sidebar .sidebar-content {{
                width: 300px;
            }}
</style>
""",
        unsafe_allow_html=True,
    )

st.write("# COVID and Economic Factors")
passw = st.text_input("Password:", value="", type="password")
if passw=='my_pass':

    df = pd.read_csv('final_df.csv')
    df.columns = [i.replace('_', ' ').title() for i in df.columns]

    st.sidebar.write('### COVID and Economic Factors')
    navigation = st.sidebar.radio('Navigation', 
                        ['Summary statistics', 'Visualizations', 'Modeling'], 0)
    
    if navigation == 'Summary statistics':
        st.subheader('Overall Info on the Data')
        show_profile = st.checkbox('Show dataset description')
        if show_profile:
            '''
            The data for the analysis ...
            '''
            profiling_table = pd.DataFrame({
                'Number of variables' : [df.shape[1]],
                'Number of observations' : [df.shape[0]],
                'Missing cells' : [df.isna().sum().sum()],
                'Missing cells (%)' : [df.isna().sum().sum()/df.shape[0]]
            })
            st.table(profiling_table)

            st.markdown('_'*100) # adding a breaking line
            st.subheader('Data Exploration')
            head_count = st.slider('How many rows of data to show?', 5, 250, 5, 5)
            which_columns = st.multiselect('Which columns to show?', df.columns.tolist(), Y_vars)
            st.dataframe(df[which_columns].head(head_count))

            st.markdown('_'*100) # adding a breaking line
            st.subheader('Summary Statistics per continent')
            col1, col2 = st.beta_columns([1,1])
            continuous_var = col1.selectbox('Continuous variable', df.columns.tolist(), 6)
            grouping_var_sum_stats = st.selectbox('Grouping variable', ['Continent', 'Sub Region'], 0)
            agg_func = col2.multiselect('Aggregation function', 
            ['mean', 'median', 'std', 'count'], ['mean', 'count'])

            sum_stats = df.groupby(grouping_var_sum_stats)[continuous_var].agg(agg_func)
            st.dataframe(sum_stats)

    if navigation == 'Visualizations':
        st.markdown('_'*100) # adding a breaking line
        col1_scatter, col2_scatter, col3_scatter = st.beta_columns(3)
        x_var_scatter = col1_scatter.selectbox('X variable (scatter)', X_vars, 1)
        y_var_scatter = col2_scatter.selectbox('Y variable (scatter)', Y_vars, 2)
        grouping_var_scatter = col3_scatter.selectbox('Color variable (scatter)', grouping_vars, 0)


        # Transformations
        x_transformation = st.selectbox('Transformatin of X variable', 
                                        ['No transformation', 'log', 'square'], 0)
        
        if x_transformation == 'No transformation':
            Y = df[y_var_scatter]
            X = df[x_var_scatter]

        elif x_transformation == 'log':
            Y = df[y_var_scatter]
            X = np.log(df[x_var_scatter])

        elif x_transformation == 'square':
            Y = df[y_var_scatter]
            X = df[x_var_scatter]**2

        X = sm.add_constant(X)
        model = sm.OLS(Y,X, missing='drop')
        results = model.fit()
        
        corr_df = df[[x_var_scatter, y_var_scatter]].dropna()
        correlation = np.corrcoef(corr_df[x_var_scatter], corr_df[y_var_scatter])[0][1]
        scatter_plot = px.scatter(df, x_var_scatter, 
                                      y_var_scatter, 
                                      color=grouping_var_scatter, 
                                      hover_data=['Country'], 
                                    #   trendline="ols",
                                      title=f'Correlation coefficient: {round(correlation, 5)}')
        
        st.plotly_chart(scatter_plot, use_container_width=True)
        st.write(results.summary())


        col1_bar, col2_bar = st.beta_columns([1, 1])
        x_var_bar = col1_bar.selectbox('X variable (histogram)', df.columns.tolist(), 6)
        grouping_var_bar = col2_bar.selectbox('Grouping variable (barplot)', grouping_vars)
        st.plotly_chart(px.histogram(df, x_var_bar, color=grouping_var_bar), use_container_width=True) 

        st.markdown('_'*100) # adding a breaking line
        col1_box, col2_box, col3_box = st.beta_columns(3)
        x_var_box = col1_box.selectbox('X variable (boxplot)', grouping_vars, 1)
        y_var_box = col2_box.selectbox('Y variable (boxplot)', df.columns.tolist(), 6)
        grouping_var_box = col3_box.selectbox('Color variable (boxplot)', grouping_vars, 0)
        st.plotly_chart(px.box(df, x_var_box, y_var_box, color=grouping_var_box), use_container_width=True)


    if navigation == 'Modeling':
        x_vars_model = st.multiselect('X variables', X_vars, ['Gdp Per Cap Ppp'])
        y_var_model = st.radio('Y variable', Y_vars, 2)
        Y = df[y_var_model]
        X = df[x_vars_model]
        X = sm.add_constant(X)
        model = sm.OLS(Y,X, missing='drop')
        results = model.fit()
        st.write('### Model Results')
        st.write(results.summary())
        st.write('### Correlation Matrix')
        st.table(df[[y_var_model]+x_vars_model].dropna().corr())



    
