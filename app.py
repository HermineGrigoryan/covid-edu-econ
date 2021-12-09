import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm


X_vars = ['gdp_growth_rate', 'gdp_per_cap_PPP',
       'personal_income_tax_rate', 'debt_to_gdp', 'ease_of_doing_business',
       'unemployment_rate', 'consumer_price_index',
       'inflation_rates', 'corruption_rank', 'population', 
       'gov_exp_on_edu_percent_of_gdp', 'gov_exp_on_edu_perc_per_student',
        'gov_exp_per_stud_tertiary_perc', 'pupil_teacher_ratio_primary',
        'youth_literacy_rate', 'gini_index']
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

st.write("## Impact of Economic and Education Factors on COVID Rates")
passw = st.text_input("Password:", value="", type="password")
if passw=='my_pass':

    df = pd.read_csv('data/final_df.csv')
    df.columns = [i.replace('_', ' ').title() for i in df.columns]

    st.sidebar.write('### COVID Analysis')
    navigation = st.sidebar.radio('Navigation', 
                        ['Exploratory Data Analysis', 'Simple Linear Regression', 'Multiple Linear Regression'], 0)
    
    if navigation == 'Exploratory Data Analysis':
        st.subheader('Overall Info on the Data')
        '''
        The aim of this project is to assess the impact of economic and educational factors of
        different countries on COVID vaccination rates.
        The data on 106 countries was obtained from [Trading Economics](https://tradingeconomics.com/)
        and [WorldBank](https://data.worldbank.org/).
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
        st.subheader('Summary Statistics per Continent & Sub-region')
        col1, col2, col3 = st.columns([1, 1, 1])
        continuous_var = col1.selectbox('Continuous variable', Y_vars+X_vars)
        grouping_var_sum_stats = col2.selectbox('Grouping variable', ['Continent', 'Sub Region'], 0)
        agg_func = col3.multiselect('Aggregation function', 
        ['mean', 'median', 'std', 'count'], ['mean', 'count'])

        sum_stats = df.groupby(grouping_var_sum_stats)[continuous_var].agg(agg_func)
        st.dataframe(sum_stats)

        st.markdown('_'*100) # adding a breaking line
        st.subheader('Visualizations')
        col1_bar, col2_bar = st.columns([1, 1])
        x_var_bar = col1_bar.selectbox('X variable (histogram)', df.columns.tolist(), 6)
        grouping_var_bar = col2_bar.selectbox('Grouping variable (barplot)', grouping_vars)
        st.plotly_chart(px.histogram(df, x_var_bar, color=grouping_var_bar), use_container_width=True) 

        st.markdown('_'*100) # adding a breaking line
        col1_box, col2_box, col3_box = st.columns(3)
        x_var_box = col1_box.selectbox('X variable (boxplot)', grouping_vars, 1)
        y_var_box = col2_box.selectbox('Y variable (boxplot)', df.columns.tolist(), 10)
        grouping_var_box = col3_box.selectbox('Color variable (boxplot)', grouping_vars, 0)
        st.plotly_chart(px.box(df, x_var_box, y_var_box, color=grouping_var_box, hover_data=['Country']), use_container_width=True)




    if navigation == 'Simple Linear Regression':
        st.markdown('_'*100) # adding a breaking line
        col1_scatter, col2_scatter, col3_scatter, col4_scatter = st.columns(4)
        x_var_scatter = col1_scatter.selectbox('X variable (scatter)', X_vars, 1)
        y_var_scatter = col2_scatter.selectbox('Y variable (scatter)', Y_vars, 2)
        grouping_var_scatter = col3_scatter.selectbox('Color variable (scatter)', grouping_vars, 0)


        # Transformations
        x_transformation = col4_scatter.selectbox('Transformatin of X variable', 
                                        ['No transformation', 'log', 'square'], 0)
        
        x_seq = np.linspace(min(df[x_var_scatter]), max(df[x_var_scatter]), 100)
        if x_transformation == 'No transformation':
            Y = df[y_var_scatter]
            X = df[x_var_scatter]
            x_seq_constant = sm.add_constant(x_seq)

        elif x_transformation == 'log':
            Y = df[y_var_scatter]
            X = np.log(df[x_var_scatter])
            x_seq_constant = sm.add_constant(np.log(x_seq))

        elif x_transformation == 'square':
            Y = df[y_var_scatter]
            X = df[x_var_scatter]**2
            x_seq_constant = sm.add_constant(x_seq**2)

        corr_df = pd.concat([X, Y], axis=1).dropna()
        correlation = np.corrcoef(corr_df[x_var_scatter], corr_df[y_var_scatter])[0][1]
        X = sm.add_constant(X)
        model = sm.OLS(Y,X, missing='drop')
        results = model.fit()
        y_pred = results.predict(x_seq_constant)
        
        scatter_plot = px.scatter(df, x_var_scatter, 
                                      y_var_scatter, 
                                      color=grouping_var_scatter, 
                                      hover_data=['Country'], 
                                      title=f'Correlation coefficient: {round(correlation, 5)}')
        
        scatter_plot.add_trace(go.Scatter(x=x_seq, y=y_pred, 
                               marker=dict(color='black'),
                               name='Prediction'))
        st.plotly_chart(scatter_plot, use_container_width=True)
        st.write(results.summary())

    if navigation == 'Multiple Linear Regression':

        # Transformations
        x_vars_no_trans = st.multiselect('X variables with no transformation', X_vars, X_vars[0])
        x_vars_log = st.multiselect('X variables with log transformation', X_vars, None)
        x_vars_square = st.multiselect('X variables with polynomial (square) transformation', X_vars, None)

        X_log = df.set_index('Country')[x_vars_log].apply(lambda x: np.log(x))
        X_log.columns = [f'{i} log' for i in X_log.columns]
        X_square = df.set_index('Country')[x_vars_square].apply(lambda x: x**2)
        X_square.columns = [f'{i} sq' for i in X_square.columns]

        if x_vars_log:
            x_vars_model = df.set_index('Country')[x_vars_no_trans].reset_index().\
                merge(X_log.reset_index()).drop('Country', axis=1)
            if x_vars_square:
                x_vars_model = df.set_index('Country')[x_vars_no_trans].reset_index().\
                merge(X_log.reset_index()).merge(X_square.reset_index()).drop('Country', axis=1)
        elif x_vars_square:
            x_vars_model = df.set_index('Country')[x_vars_no_trans].reset_index().\
                merge(X_square.reset_index()).drop('Country', axis=1)
            if x_vars_log:
                x_vars_model = df.set_index('Country')[x_vars_no_trans].reset_index().\
                merge(X_log.reset_index()).merge(X_square.reset_index()).drop('Country', axis=1)
        else:
            x_vars_model = df[x_vars_no_trans]
        y_var_model = st.radio('Y variable', Y_vars, 2)
        log_y = st.checkbox('Use log of Y instead of Y')
        if log_y:
            Y = np.log(df[y_var_model])
        else:
            Y = df[y_var_model]
        X = x_vars_model
        X = sm.add_constant(X)
        model = sm.OLS(Y,X, missing='drop')
        results = model.fit()
        st.write('### Model Results')
        st.write(results.summary())
        st.write('### Correlation Matrix')
        corr_df = x_vars_model
        corr_df[y_var_model] = Y
        corr_mat = corr_df.dropna().corr()
        st.plotly_chart(px.imshow(corr_mat), use_container_width=True)



    
