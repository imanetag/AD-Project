# Importing necessary libraries
import ssl
import urllib.request
from urllib.parse import urlparse
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.graph_objects as go 
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.stats import  poisson, norm, expon, binom, bernoulli, uniform, ks_2samp
from scipy.stats import ttest_ind, zscore, chi2_contingency, linregress, f_oneway
from statsmodels.formula.api import ols
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy.stats import gaussian_kde
# Disable the deprecated warning from Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

def read_file(uploaded_file, separator, sheet_name):
    try:
        # Check the file type and read accordingly
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=separator, quotechar='"', engine='python', encoding='utf-8')
        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        elif uploaded_file.name.lower().endswith('.txt'):
            df = pd.read_fwf(uploaded_file)
        else:
            # Display an error message for unsupported file formats
            st.error('Unsupported file format')
            return None
        # Display the original DataFrame in the Streamlit app
        st.write("Original DataFrame:")
        st.write(df)
        # Return the DataFrame for further processing
        return df
    
    # Handle specific exceptions
    except pd.errors.EmptyDataError:
        st.warning("The file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("Error reading file. Please check the file format.")
        return None
    except Exception as e:
        # Display a general error message for any other exceptions        
        st.error(f"Error reading file: {str(e)}")
        return None

def read_data_from_link(link, separator, sheet_name):
    # Check if a link is provided
    if link:
        try:
            # Disable SSL certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            # Open the link and get the file extension
            response = urllib.request.urlopen(link, context=ssl_context)
            file_extension = urlparse(link).path.split('.')[-1]
            # Print file information for debugging purposes
            print(f"File link: {link}, File Extension: {file_extension}")
            # Read the content based on file extension
            if file_extension.lower() == 'csv':
                content = response.read().decode('latin-1')
                df = pd.read_csv(StringIO(content), sep=separator)
            elif file_extension.lower() in ('xls', 'xlsx'):
                content = response.read().decode('latin-1')
                df = pd.read_excel(StringIO(content), sheet_name=sheet_name)
            elif file_extension.lower() == 'txt':
                content = response.read().decode('latin-1')
                df = pd.read_fwf(StringIO(content))
            else:
                # Display an error message for unsupported file extensions
                st.error(f"Unsupported file extension for link: {file_extension}")
                return None
            # Return the DataFrame for further processing
            return df
        # Handle specific exceptions
        except pd.errors.EmptyDataError:
            st.warning("The file from the link is empty.")
            return None
        except Exception as e:
            # Display a general error message for any other exceptions
            st.error(f"Error loading data from link: {e}")
            return None
    else:
        # Return None if no link is provided
        return None

def calculate_data(st, df_raw, operation, selected_columns=None):
    try:
        # Perform different operations based on the user's choice
        if operation == "describe":
            # Return descriptive statistics for the DataFrame
            return df_raw.describe(include='all')
        elif operation == "tail":
            # Return the last n rows of the DataFrame
            return df_raw.tail()
        elif operation == "head":
            # Return the first n rows of the DataFrame
            return df_raw.head()
        elif operation == "info":
    # Capture and display DataFrame information
            buffer = StringIO()
            df_raw.info(buf=buffer)
            info_output = buffer.getvalue()
    # Display the captured info using st.text
            st.text(info_output)
            return None 
        elif operation == "null_values":
            # Check if selected_columns is provided
            if selected_columns:
                df_raw = df_raw[selected_columns]
            # Return a DataFrame showing null values count for each column
            result = pd.DataFrame({
                "Column": df_raw.columns,
                "Null Values": df_raw.isnull().sum().values
            })
            return result
        elif operation == "drop_nulls":
            # Drop rows with null values and return the new DataFrame
            return df_raw.dropna()
        elif operation == "unique_values":
            if selected_columns:
                # Return unique values for the selected column
                return pd.DataFrame({selected_columns[0]: df_raw[selected_columns[0]].unique()})
            else:
                # Display a warning if no column is selected
                st.warning("Please select a column for unique values.")
                return st.warning("Please select a column for unique values.")

        elif operation == "value_counts":
            if selected_columns:
                # Return value counts for the selected column
                return df_raw[selected_columns[0]].value_counts().reset_index()
            else:
                # Display a warning if no column is selected
                st.warning("Please select a column for value counts.")
                return st.warning("Please select a column for value counts.")
        elif operation in ["Maximum", "Minimum", "Sum", "Count"]:
            # Prompt the user to select a column for numeric operations
            selected_column = st.selectbox("Select a column:", df_raw.columns.tolist(), key="column_selector")
            if selected_column is None:
                return st.warning("Please select a column for the operation.")

            # Perform the operation on the selected column
            if pd.api.types.is_numeric_dtype(df_raw[selected_column]):
                if operation == "Maximum":
                    result = pd.DataFrame({operation: [df_raw[selected_column].max()]})
                    st.write(result)
                elif operation == "Minimum":
                    result = pd.DataFrame({operation: [df_raw[selected_column].min()]})
                    st.write(result)
                elif operation == "Sum":
                    if df_raw[selected_column].dtype == bool:
                        result = pd.DataFrame({operation: [df_raw[selected_column].sum()]})
                        st.warning(f"The selected column '{selected_column}' is boolean. Showing the sum of boolean values after conversion.")
                        st.write(result)
                    else:
                        st.warning(f"The selected column '{selected_column}' is numeric. Showing the sum of numeric values.")
                        result = pd.DataFrame({operation: [df_raw[selected_column].sum()]})
                        st.write(result)
            elif df_raw[selected_column].dtype == bool:
                # Handle boolean columns
                if operation == "Maximum":
                    result = st.warning(f"The selected column '{selected_column}' is boolean. Showing the maximum value as True (1).")
                    st.write(result)
                elif operation == "Minimum":
                    result = st.warning(f"The selected column '{selected_column}' is boolean. Showing the minimum value as False (0).")
                    st.write(result)
            else:
                # Handle non-numeric data differently
                st.warning(f"The selected column '{selected_column}' is not numeric. Cannot perform '{operation}' operation.")
            return None
    except Exception as e:
        # Handle exceptions and display an error message
        st.error(f"Error calculating data: {str(e)}")
        return None

def indexation_section(df):
    # Display a subheader for the indexation section
    st.subheader("Indexation:")
    # Allow the user to choose an indexing option
    indexing_option = st.radio("Choose indexing option:", ("Index Columns", "Index Rows", "Index Value"))
    # Perform different actions based on the chosen indexing option
    if indexing_option == "Index Columns":
        # Allow the user to select columns for indexing
        selected_columns = st.multiselect("Choose columns for indexing:", df.columns)
        if selected_columns:
            st.table(df[selected_columns])
        else:
            st.warning("Please select columns for indexing.")

    elif indexing_option == "Index Rows":
        # Call the function to handle indexing by rows
        handle_index_rows(df)

    elif indexing_option == "Index Value":
        # Call the function to handle indexing by a specific value
        handle_index_value(df)

    else:
        st.warning("Please choose a valid indexing option.")

def handle_index_rows(df):
    # Allow the user to choose row indices for indexing
    index_row_selector = st.multiselect("Choose row indices:", df.index, key='index_rows_selector')

    # Display the table dynamically based on the selected row indices
    if index_row_selector:
        try:
            # Transpose the DataFrame for displaying rows as columns
            filtered_df = df.loc[index_row_selector]
            st.table(filtered_df)
        except KeyError:
            st.warning("No rows found for the selected indices.")
    else:
        st.warning("Please choose at least one valid row index for indexing.")

def handle_index_value(df):
    # Allow the user to choose a column and row index for indexing by value
    with st.form(key='index_value_form'):
        index_column_value = st.selectbox("Choose column for indexing:", df.columns, key='index_value_column_selector')
        index_row_value = st.selectbox("Choose row index:", df.index, key='index_value_row_selector')
        submit_value = st.form_submit_button(label='Submit Value')

        # Handle the form submission
        if submit_value:
            handle_submit_value(df, index_column_value, index_row_value)

def handle_submit_value(df, index_column_value, index_row_value):
    # Check if the chosen row and column indices are valid
    if index_column_value in df.columns and index_row_value in df.index:
        try:
            # Get and display the value at the specified row and column
            value = df.at[index_row_value, index_column_value]
            st.write(f"Value at row {index_row_value} and column {index_column_value}: {value}")
        except KeyError:
            st.warning(f"No value found for row {index_row_value} and column {index_column_value}.")
    else:
        st.warning("Please choose valid row and column indices for indexing.")

def calculate_operation(df, operation, selected_columns=None):
    try:
        # If no columns are explicitly selected, use all columns in the DataFrame
        if selected_columns is None:
            selected_columns = df.columns
        # Initialize a dictionary to store results for each column
        result_dict = {}
        
        # Iterate through selected columns
        for column in selected_columns:
            # Check if the column has boolean values
            if df[column].dtype == bool:
                result_dict[column] = f"No {operation} found (Boolean column)"
            else:
                # Check if the column has numeric values
                if pd.to_numeric(df[column], errors='coerce').notna().any():
                    # Convert selected column to numeric type
                    numeric_column = pd.to_numeric(df[column], errors='coerce')
                    # Perform the selected operation
                    result = None
                    if operation == "Median":
                        result = numeric_column.median(skipna=True)
                    elif operation == "Mean":
                        result = numeric_column.mean(skipna=True)
                    elif operation == "Mode":
                        result = custom_mode(numeric_column)
                    elif operation == "Variance":
                        result = numeric_column.var(skipna=True)
                    elif operation == "Standard Deviation":
                        result = numeric_column.std(skipna=True)
                    elif operation == "Range":
                        result = numeric_column.max() - numeric_column.min()
                    # Format the result
                    result_dict[column] = f"No {operation} found" if result is None or pd.isna(result) else result
                else:
                    # Non-numeric column
                    result_dict[column] = f"No {operation} found (Non-numeric column)"
        # Create a DataFrame with the result
        result_df = pd.DataFrame({operation: result_dict}).reset_index()
        return result_df
    except Exception as e:
        # Handle exceptions and print an error message
        print(f"Error calculating operation: {str(e)}")
        return None

def custom_mode(column):
    try:
        # Calculate the mode(s) of the column
        modes = column.mode()

        # Check if mode(s) exist
        if not modes.empty:
            # Return the mode(s) as a comma-separated string
            return ', '.join(map(str, modes))

        # Return None if no mode(s) found
        return None
    except Exception as e:
        # Handle exceptions and print an error message
        print(f"Error in custom_mode: {e}")
        return None

def calculate_statistics(df, operation, selected_columns=None):
    try:
        # Convert selected columns to numeric
        if selected_columns is None:
            selected_columns = df.columns

        # Apply numeric conversion to selected columns, handling errors with 'coerce'
        Selected_Columns = df[selected_columns].apply(pd.to_numeric, errors='coerce')

        # Convert boolean columns to int for skewness calculation
        for column in selected_columns:
            if df[column].dtype == 'bool':
                Selected_Columns[column] = Selected_Columns[column].astype(int)

        if operation == "Skewness":
        # Calculate skewness
            result = Selected_Columns.skew()

        # Display the skewness
            #st.write(f"Skewness: {result}")

        # Plot the skewness with legend
            for column in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax, label="Data Distribution")
                ax.set_title(f'Distribution of {column}')
                ax.axvline(Selected_Columns[column].skew(), color='red', linestyle='dashed', linewidth=2, label="Skewness Mean")
                ax.legend()
                st.pyplot(fig)

        # Handle NaN values
            result_df = pd.DataFrame({operation: result}, index=selected_columns)
            result_df = result_df.map(lambda x: 'error' if pd.isna(x) or np.isnan(x) else str(x))

        elif operation == "Kurtosis":
            result = Selected_Columns.kurtosis()
            
            # Display the kurtosis
            #st.write(f"Kurtosis: {result}")
            
            # Plot the distribution with legend
            for column in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax, label="Data Distribution")
                ax.set_title(f'Distribution of {column}')
                ax.axvline(Selected_Columns[column].kurtosis(), color='red', linestyle='dashed', linewidth=2, label="Kurtosis Mean")
                ax.legend()
                st.pyplot(fig)
                
            # Handle NaN values
            result_df = pd.DataFrame({operation: result}, index=selected_columns)
            result_df = result_df.map(lambda x: 'error' if pd.isna(x) or np.isnan(x) else str(x))

        elif operation == "Cumulative Sum":
            # Calculate cumulative sum
            result = Selected_Columns.cumsum()
            # Handle NaN values
            result_df = result.copy()
            result_df[selected_columns] = result_df[selected_columns].fillna('error')

        elif operation == "Correlation":
            # Calculate correlation
            result = Selected_Columns.corr()
            # Handle NaN values
            result_df = result.copy()
            result_df = result_df.map(lambda x: 'error' if pd.isna(x) or np.isnan(x) else x)

        elif operation == "Covariance":
            # Calculate covariance
            result = Selected_Columns.cov()
            # Handle NaN values
            result_df = result.copy()
            result_df = result_df.map(lambda x: 'error' if pd.isna(x) or np.isnan(x) else x)

        elif operation == "Quantiles":
            # Calculate quantiles (25%, 50%, 75%)
            result = Selected_Columns.quantile([0.25, 0.5, 0.75])
            # Handle NaN values          
            result_df = result.copy()
            #result_df = result_df.applymap(lambda x: 'error' if pd.isna(x) or np.isnan(x) else x)
            result_df = result_df.apply(lambda x: x.map(lambda y: 'error' if pd.isna(y) or np.isnan(y) else y))

        elif operation == "Absolute Change":
            # Calculate absolute change
            result = Selected_Columns.diff().abs()
            result.iloc[0, :] = Selected_Columns.iloc[0, :]  # Set the first row to the original values
            # Handle NaN values
            result_df = result.copy()
            result_df = result_df.map(lambda x: 'error' if pd.isna(x) or np.isnan(x) else x)

        elif operation == "Cumulative Product":
            # Calculate cumulative product
            result = Selected_Columns.cumprod()
            # Handle NaN values
            result_df = result.copy()
            result_df = result_df[selected_columns].fillna('error') 

        elif operation == "IQR":
            # Calculate interquartile range (IQR)
            q3 = Selected_Columns.quantile(0.75)
            q1 = Selected_Columns.quantile(0.25)
            result = q3 - q1
            # Handle NaN values
            result_df = result.copy()
            result_df = result_df.apply(lambda x: 'error' if pd.isna(x) or np.isnan(x) else x)
            #result_df = result_df.apply(lambda x: x.map(lambda y: 'error' if pd.isna(y) or np.isnan(y) else y))
            
        return result_df
            
    except Exception as e:
        # Handle exceptions and display an error message
        st.error(f"Error calculating statistics: {str(e)}")
        return None

def visualize_data(df, visualization_type, selected_columns, distribution_type=None):
    try:
        # Display information about the chosen visualization type and selected columns
        st.write(f"Visualization Type: {visualization_type}")
        st.write(f"Selected Columns: {selected_columns}")
        
        if visualization_type == "Line Plot":
            # Check the number of selected columns for Line Plot
            if len(selected_columns) == 1:
                y_variable = selected_columns[0]
                # Check if the selected column is present in the DataFrame and contains numeric data
                if y_variable in df.columns and pd.api.types.is_numeric_dtype(df[y_variable]):
                # Create and display a Line Plot for the selected column
                    fig = px.line(df, y=y_variable, title=f"Line Plot for {y_variable}")
                    st.plotly_chart(fig)
                elif y_variable in df.columns:
                    st.warning(f"Selected column '{y_variable}' does not contain numeric data. Please choose a numeric column.")
                else:
                    st.warning("Please select a valid column for Line Plot.")
            elif len(selected_columns) == 2:
                x_variable, y_variable = selected_columns
                # Sort DataFrame by x_variable
                df_sorted = df.sort_values(by=x_variable)
                # Check if both selected columns are present in the DataFrame and contain numeric data
                if x_variable in df.columns and y_variable in df.columns and \
                        pd.api.types.is_numeric_dtype(df[x_variable]) and pd.api.types.is_numeric_dtype(df[y_variable]):
                    # Create and display a Line Plot for the selected columns
                    fig = px.line(df_sorted, x=x_variable, y=y_variable, title=f"Line Plot for {x_variable} and {y_variable}")
                    st.plotly_chart(fig)
                elif x_variable in df.columns and y_variable in df.columns:
                    st.warning(f"Selected columns '{x_variable}' and '{y_variable}' must contain numeric data for Line Plot.")
                else:
                    st.warning("Please select valid columns for Line Plot.")
            else:
                st.warning("Please select one or two columns for Line Plot.")
        elif visualization_type == "Scatter Plot":
            # Check the number of selected columns for Scatter Plot
            if len(selected_columns) == 2:
                x_variable = selected_columns[0]
                y_variable = selected_columns[1]
                
                # Check if both selected columns are numeric
                if not pd.api.types.is_numeric_dtype(df[x_variable]) or not pd.api.types.is_numeric_dtype(df[y_variable]):
                    st.warning("Please select numeric columns for Scatter Plot.")
                else:
                    # Create and display a Scatter Plot for the selected columns
                    fig = px.scatter(df, x=x_variable, y=y_variable, title="Scatter Plot")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select two columns for Scatter Plot.")

        elif visualization_type == "Box Plot":
            # Check if at least one column is selected for Box Plot
            if selected_columns:
                # Check if all selected columns are numeric
                if all(pd.api.types.is_numeric_dtype(df[column]) for column in selected_columns):
                    # Melt the DataFrame to reshape it for a single box plot
                    df_melted = df[selected_columns].melt(var_name='Variable', value_name='Value')
                    # Create a box plot with the melted DataFrame
                    fig = px.box(df_melted, x='Variable', y='Value', title="Box Plot")
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select numeric columns for Box Plot.")
            else:
                st.warning("Please select at least one column for Box Plot.")

        elif visualization_type == "Histogram":
            # Check if exactly one column is selected for Histogram
            if len(selected_columns) == 1:
                x_variable = selected_columns[0]
                # Check if the selected column is present in the DataFrame
                if x_variable in df.columns:
                    # Check if the selected column is numeric
                    if pd.api.types.is_numeric_dtype(df[x_variable]):
                        fig = px.histogram(df, x=x_variable, title="Histogram")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select a numeric column for Histogram.")
                else:
                    st.warning("Please select a valid column for Histogram.")
            else:
                st.warning("Please select only one column for Histogram.")

        elif visualization_type == "KDE Plot":
            # Check if exactly one column is selected for KDE Plot
            if len(selected_columns) == 1:
                selected_column = selected_columns[0]
                # Check if the selected column is numeric
                if not pd.api.types.is_numeric_dtype(df[selected_column]):
                    st.warning(f"The selected column '{selected_column}' is not numeric. Please choose a numeric column for KDE Plot.")
                else:
                    # Display a subheader for the KDE Plot section
                    st.subheader("Kernel Density Estimation (KDE) Plot:")

                    # Calculate KDE using scipy
                    kde = gaussian_kde(df[selected_column])
                    x_vals = np.linspace(df[selected_column].min(), df[selected_column].max(), 1000)
                    y_vals = kde(x_vals)

                    # Create KDE plot using Plotly Graph Objects
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(width=2, color='blue'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.4)', name='KDE')
                    )

                    # Customize layout
                    fig.update_layout(
                        xaxis_title=selected_column,
                        yaxis_title="Density",
                        title="Kernel Density Estimation (KDE) Plot",
                    )

                    # Show the plot
                    st.plotly_chart(fig)

                    # Identify the distribution after displaying the KDE Plot
                    distribution_type = detect_distribution_type(df[selected_column])
                    st.write(f"This KDE plot may suggest a {distribution_type} distribution.")
            else:
                st.warning("Please select exactly one column for KDE Plot.")
        elif visualization_type == "Bar Plot":
            # Check if exactly two columns are selected for Bar Plot
            if len(selected_columns) == 2:
                x_variable, y_variable = selected_columns

                # Check if the selected columns are valid
                if x_variable in df.columns and y_variable in df.columns:
                    # Display data types and the selected data for exploration
                    st.write(df.dtypes[[x_variable, y_variable]])            
                    st.write(df[[x_variable, y_variable]])

                    # Check if the Y-axis variable is categorical
                    #if pd.api.types.is_categorical_dtype(df[y_variable]):
                    if isinstance(df[y_variable].dtype, pd.CategoricalDtype):
                        category_orders = {
                            y_variable: df[y_variable].cat.categories.tolist()
                        }
                        fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable, category_orders=category_orders)
                    else:
                        # Check if the X-axis variable is categorical
                        #if pd.api.types.is_categorical_dtype(df[x_variable]):
                        if isinstance(df[x_variable].dtype, pd.CategoricalDtype):
                            category_orders = {
                                x_variable: df[x_variable].cat.categories.tolist()
                            }
                            fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable, category_orders=category_orders)
                        else:
                            fig = px.bar(df, x=x_variable, y=y_variable, title="Bar Plot", color=x_variable)

                    st.plotly_chart(fig)
                else:
                    st.warning("Please select valid columns for Bar Plot.")
            else:
                st.warning("Please select two columns for Bar Plot.")

        elif visualization_type == "Heatmap1":
            # Check if at least two columns are selected for Heatmap
            if len(selected_columns) >= 2:
                # Check if the selected columns are present in the DataFrame
                if all(column in df.columns for column in selected_columns):
                    selected_data = df[selected_columns].copy()

                    # Encode categorical columns to numerical values
                    for column in selected_data.select_dtypes(include='object').columns:
                        label_encoder = LabelEncoder()
                        #selected_data[column] = label_encoder.fit_transform(selected_data[column])
                        selected_data.loc[:, column] = label_encoder.fit_transform(selected_data[column])


                    corr_matrix = selected_data.corr()
                    # Create a heatmap with correlation values using Plotly
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='Viridis',
                        labels=dict(x="X-axis", y="Y-axis", color="Correlation"),
                        height=800,
                        width=1000,
                    )
                    # Add text annotations
                    for i, row in enumerate(corr_matrix.index):
                        for j, col in enumerate(corr_matrix.columns):
                            fig.add_annotation(
                                x=col,
                                y=row,
                                text=str(corr_matrix.iloc[i, j].round(2)),
                                showarrow=False,
                                font=dict(color='black'),
                            )

                    # Customize layout
                    fig.update_layout(
                        title="Heatmap with Correlation Values",
                        xaxis_title="X-axis",
                        yaxis_title="Y-axis",
                    )

                    st.plotly_chart(fig)
                else:
                    st.warning("Please select valid columns for Heatmap.")
            else:
                st.warning("Please select at least two columns for Heatmap.")
        elif visualization_type == "Heatmap2":
            # Select only numeric columns
            numeric_columns = df.select_dtypes(include=['number'])

            # Encode categorical columns to numerical values
            for column in df.select_dtypes(include='object').columns:
                label_encoder = LabelEncoder()
                df[column] = label_encoder.fit_transform(df[column])

            corr_matrix = numeric_columns.corr()

            # Convert NaN values to a string representation
            #corr_matrix_string = corr_matrix.applymap(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            corr_matrix_string = corr_matrix.apply(lambda x: x.map(lambda y: f"{y:.2f}" if not pd.isna(y) else "N/A"))

            fig, ax = plt.subplots(figsize=(10, 8))
            # Create a heatmap with correlation values using Seaborn
            sns.heatmap(
                corr_matrix,
                annot=corr_matrix_string,
                cmap="viridis",
                fmt="",
                linewidths=.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )

            ax.set_title("Heatmap with Correlation Values")
            st.pyplot(fig)

        elif visualization_type == "Pie Chart":
            # Check if exactly one column is selected for Pie Chart
            if len(selected_columns) == 1:
                selected_column = selected_columns[0]

                # Check if the selected column is categorical
                #if pd.api.types.is_categorical_dtype(df[selected_column]):
                if isinstance(df[selected_column].dtype, pd.CategoricalDtype):
                    unique_values = len(df[selected_column].cat.categories)
                    if unique_values > 20:
                        st.warning("Too many unique values for Pie Chart. Select a column with fewer categories.")
                        return
                # Create a Pie Chart using Plotly
                fig = px.pie(df, names=selected_column, title=f"Pie Chart - {selected_column}")
                st.plotly_chart(fig)
            else:
                st.warning("Please select one column for Pie Chart.")

        elif visualization_type == "Violin Plot":
            
            # Check if at least one column is selected for Violin Plot
            if len(selected_columns) >= 1:
                # If one column is selected, create Violin Plot with y as the selected column
                if len(selected_columns) == 1:
                    y_variable = selected_columns[0]
                    if y_variable in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Violin(y=df[y_variable], box_visible=True, line_color='blue', fillcolor='lightblue', name=y_variable))
                        fig.update_layout(title=f"Violin Plot for {y_variable}", showlegend=True)
                        st.plotly_chart(fig)
                        st.markdown(f"***Axe Y :*** {y_variable}")
                    else: 
                        st.warning("Please select a valid column for Violin Plot.")
                # If two columns are selected, create Violin Plot with x and y as the selected columns
                elif len(selected_columns) == 2:
                    x_variable =selected_columns[0] 
                    y_variable =selected_columns[1] 
                    if x_variable in df.columns and y_variable in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Violin(x=df[x_variable], y=df[y_variable], box_visible=True, line_color='blue', fillcolor='lightblue', name=y_variable))
                        fig.update_layout(title=f"Violin Plot for {x_variable} and {y_variable}", showlegend=True)
                        st.plotly_chart(fig)
                        st.markdown(f"***Axe X :*** {x_variable}")
                        st.markdown(f"***Axe Y :*** {y_variable}")
                    else:
                        st.warning("Please select valid columns for Violin Plot.")
                else:
                    st.warning("Please select only one or two columns for Violin Plot.")
            else:
                st.warning("Please select at least one column for Violin Plot.")
        elif visualization_type == "Pair Plot":
            try:
                # Check if at least two columns are selected for Pair Plot
                if len(selected_columns) >= 2:
                    # Filter numeric columns
                    numeric_columns = df[selected_columns].select_dtypes(include=['number'])
                    if not numeric_columns.empty:
                        # Create a Pair Plot using Plotly
                        fig = px.scatter_matrix(numeric_columns, title="Pair Plot")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least two numeric columns for Pair Plot.")
                else:
                    st.warning("Please select at least two columns for Pair Plot.")
            except Exception as e:
                st.error(f"Error visualizing Pair Plot: {str(e)}")
                return None
            
        elif not visualization_type:
            st.warning("Please choose a valid visualization type.")
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
        return None
    
# Dictionary mapping visualization types to numerical identifiers
visualization_types = {
    "Line Plot": 1,
    "Bar Plot": 2,
    "Box Plot": 3,
    "Histogram": 4,
    "Scatter Plot": 5,    
    "Pie Chart": 6,
    "Heatmap1": 7,
    "Heatmap2": 8,
    "Violin Plot": 9,
    "KDE Plot": 10,
    "Pair Plot": 11,
                 }
def detect_distribution_type(data, kde=None):
    try:
        if kde is not None:
            # Use KDE to determine distribution type
            kde_plot = kde.evaluate(np.linspace(np.min(data), np.max(data), 1000))
            peaks, _ = find_peaks(kde_plot)
            if len(peaks) > 1:
                detected_distribution_types = ["Bimodal"]
            else:
                detected_distribution_types = []  # Keep empty if distribution type is not confidently identified
        else:
            # Use existing methods if KDE is not available
            detected_distribution_types = []
            
            # Poisson
            try:
                # Fit Poisson distribution to data
                poisson_params = (np.round(data.mean()),)  # Convert mean to integer for Poisson distribution
                poisson_cdf = poisson.cdf(np.round(data), *poisson_params)
    
                # Calculate KS statistic
                ks_stat_poisson, _ = ks_2samp(data, poisson_cdf)
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Poisson", ks_stat_poisson))
            except Exception:
                pass

            try:
                # Fit Normal distribution to data
                params_normal = norm.fit(data)
    
                # Calculate KS statistic
                ks_stat_normal, _ = ks_2samp(data, norm.cdf(data, *params_normal))
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Normal", ks_stat_normal))
            except Exception:
                pass

            try:
                # Fit Exponential distribution to data
                params_exponential = expon.fit(data)
    
                # Calculate KS statistic
                ks_stat_exponential, _ = ks_2samp(data, expon.cdf(data, *params_exponential))
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Exponential", ks_stat_exponential))
            except Exception:
                pass


            try:
                # Fit Binomial distribution to data
                params_binomial = (data.sum(), len(data), 0.5)
    
                # Convert data to integers before calculating CDF
                data_int = np.round(data).astype(int)
    
                # Calculate KS statistic
                ks_stat_binomial, _ = ks_2samp(data_int, binom.cdf(data_int, *params_binomial))
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Binomial", ks_stat_binomial))
            except Exception:
                pass

            try:
                # Fit Bernoulli distribution to data
                params_bernoulli = np.mean(data)
    
                # Convert data to 0 or 1 (Bernoulli values)
                data_bernoulli = (data >= params_bernoulli).astype(int)
    
                # Calculate KS statistic
                ks_stat_bernoulli, _ = ks_2samp(data_bernoulli, bernoulli.cdf(data_bernoulli, params_bernoulli))
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Bernoulli", ks_stat_bernoulli))
            except Exception:
                pass

            try:
                # Fit Uniform distribution to data
                params_uniform = np.min(data), np.max(data)
    
                # Calculate KS statistic using uniform distribution CDF
                ks_stat_uniform, _ = ks_2samp(data, uniform.cdf(data, *params_uniform))
    
                # Append result to the list of detected distribution types
                detected_distribution_types.append(("Uniform", ks_stat_uniform))
            except Exception:
                pass

            # List of distributions to check using statistical tests 
            distributions_to_check = [
                (norm, "Normal"),
                (expon, "Exponential"),
                (binom, "Binomial"),
                (bernoulli, "Bernoulli"),
                (uniform,"Uniform"),
                (poisson,"Poisson"), 
            ]

            for distribution, distribution_name in distributions_to_check:
                try:
                    # Fit distribution to data
                    params = distribution.fit(data)
            
                    # Calculate KS statistic using distribution CDF
                    ks_stat, _ = ks_2samp(data, distribution.cdf(data, *params))
            
                    # Append result to the list of detected distribution types
                    detected_distribution_types.append((distribution_name, ks_stat))
                except Exception:
                    pass
                
        # Sort by KS statistic in descending order
        detected_distribution_types.sort(key=lambda x: x[1], reverse=True)
    
        if detected_distribution_types:
            detected_distribution_type = detected_distribution_types[0][0]
            st.write(f"Detected Distribution Type: {detected_distribution_type}")
            return detected_distribution_type
        else:
            st.warning("Distribution type could not be confidently identified.")
            return None
        
    except Exception as e:
        st.error(f"Error detecting distribution type: {str(e)}")
        return None
def perform_t_test(st, df, column1, column2):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2])):
            st.error("Please select numeric columns for t-test.")
            return

        # Perform independent t-test
        t_statistic, p_value = ttest_ind(df[column1], df[column2])

        # Display results
        st.write(f"Independent t-test results for '{column1}' and '{column2}':")
        st.write(f"T-statistic: {t_statistic}")
        st.write(f"P-value: {p_value}")

        # Interpret the results (you can customize this part based on your needs)
        alpha = 0.05
        if p_value < alpha:
            st.write("The difference between groups is statistically significant.")
        else:
            st.write("There is no significant difference between groups.")
    except Exception as e:
        st.error(f"Error performing t-test: {str(e)}")

def perform_z_test(st, df, column1, column2=None, alpha=0.05):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[column1]) and (column2 is None or pd.api.types.is_numeric_dtype(df[column2]))):
            st.error("Please select numeric columns for the z-test.")
            return

        # Perform z-test
        if column2 is None:
            # One-sample z-test
            z_scores = zscore(df[column1])
            sample_mean = np.mean(df[column1])
        else:
            # Two-sample z-test
            z_scores = zscore(df[column1] - df[column2])
            sample_mean = np.mean(df[column1] - df[column2])

        # Calculate P-values
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        # Display results
        st.subheader("Z Test Results")

        st.write(f"Z-test results for '{column1}'" + (f" and '{column2}'" if column2 else "") + ":")
        st.write(f"Sample Mean: {sample_mean}")
        st.write(f"Z-scores: Min = {np.min(z_scores)}, Max = {np.max(z_scores)}")
        st.write(f"P-Values: Min = {np.min(p_values)}, Max = {np.max(p_values)}")

        # Evaluate significance
        significant = any(p_value < alpha for p_value in p_values)
        significance_message = "There is a significant association between the variables" if significant else "There is no significant association between the variables"
        st.write(significance_message)

        # Display individual results
        st.write("Individual Results:")
        for i, (p_val, z_stat_val) in enumerate(zip(p_values, z_scores)):
            st.write(f"Pair {i + 1}: P-Value = {p_val}, Z-Statistic = {z_stat_val}")

    except Exception as e:
        st.error(f"Error performing and displaying Z-test: {str(e)}")


def perform_chi_square_test(st, df, column1, column2):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2])):
            st.error("Please select numeric columns for the chi-square test.")
            return

        # Perform chi-square test
        contingency_table = pd.crosstab(df[column1], df[column2])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

        # Display results
        st.write(f"Chi-square test results for '{column1}' and '{column2}':")
        st.write(f"Chi2 statistic: {chi2_stat}")
        st.write(f"P-value: {p_value}")

        # Interpret the results (you can customize this part based on your needs)
        alpha = 0.05
        if p_value < alpha:
            st.write("There is a significant association between the variables.")
        else:
            st.write("There is no significant association between the variables.")
    except Exception as e:
        st.error(f"Error performing chi-square test: {str(e)}")


def perform_linear_regression(st, df, x_column, y_column):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column])):
            st.error("Please select numeric columns for linear regression.")
            return

        # Check for sufficient data
        if df[x_column].notnull().sum() < 2 or df[y_column].notnull().sum() < 2:
            st.warning("Insufficient data for linear regression. Please select columns with at least two non-null values.")
            return

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df[x_column], df[y_column])

        # Display results
        st.write(f"Linear Regression results for '{x_column}' and '{y_column}':")
        st.write(f"Slope: {slope}")
        st.write(f"Intercept: {intercept}")
        st.write(f"R-value: {r_value}")
        st.write(f"P-value: {p_value}")
        st.write(f"Standard Error: {std_err}")

        # Display the equation of the regression line
        equation = f"y = {slope:.4f} * x + {intercept:.4f}"
        st.write(f"Equation of the regression line: {equation}")

        # Plot regression line and scatter plot
        plt.scatter(df[x_column], df[y_column], label="Data Points")
        plt.plot(df[x_column], intercept + slope * df[x_column], color='red', label='Regression Line')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error performing linear regression: {str(e)}")
    
def one_way_anova(df, group_col, target_col):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[group_col]) and pd.api.types.is_numeric_dtype(df[target_col])):
            raise ValueError("Please select numeric columns for one-way ANOVA.")

        # Perform one-way ANOVA
        groups = [df[df[group_col] == value][target_col] for value in df[group_col].unique()]
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value

    except Exception as e:
        raise ValueError(f"Error performing one-way ANOVA: {str(e)}") from e

def perform_one_way_anova(st, df, group_column, value_column):
    try:
        # Call the one_way_anova function
        f_statistic, p_value = one_way_anova(df, group_column, value_column)

        # Display results
        st.write(f"One-way ANOVA results for '{value_column}' across '{group_column}':")
        st.write(f"F-statistic: {f_statistic}")
        st.write(f"P-value: {p_value}")

        # Evaluate significance
        significance_message = "Significant difference detected." if p_value < 0.05 else "No significant difference detected."
        st.write(significance_message)

    except ValueError as ve:
        st.error(str(ve))
    
# Function for two-way ANOVA
def two_way_anova(st, df, group_column1, group_column2, value_column):
    try:
        # Check if columns are numeric
        if not (pd.api.types.is_numeric_dtype(df[group_column1]) and pd.api.types.is_numeric_dtype(df[group_column2]) and pd.api.types.is_numeric_dtype(df[value_column])):
            st.error("Please select numeric columns for two-way ANOVA.")
            return

        # Perform two-way ANOVA
        formula = f"{value_column} ~ {group_column1} * {group_column2}"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Display results
        st.write(f"Two-way ANOVA results for '{value_column}' across '{group_column1}' and '{group_column2}':")
        st.write(anova_table)

        # Print summary to console for debugging
        print(anova_table)

        # Evaluate significance
        interaction_term = f"{group_column1}:{group_column2}"
        p_value = anova_table.loc[interaction_term, 'PR(>F)']

        significance_message = "Significant difference detected." if p_value < 0.05 else "No significant difference detected."
        st.write(significance_message)

    except Exception as e:
        st.error(f"Error performing two-way ANOVA: {str(e)}")

        
def main():
    
    # Set the title for the web app
    st.title("Data Analysis Web App Project")
    # Set display options for pandas to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Allow the user to choose between uploading a file or providing a link
    file_or_link = st.radio("Choose data input method:", ("File Upload", "Link Input"))
    df_raw = None
    # Allow the user to choose a separator for the data (default options include ",", ";", "/", and space)
    separator = st.selectbox("Choose a separator", [",", ";", "/"," "], key="separator")

    # If the user chooses "File Upload"
    if file_or_link == "File Upload":
        # Allow the user to upload a file
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "xls", "xlsx"])

        if uploaded_file is not None:
            st.write("File uploaded successfully!")
            # If the uploaded file is an Excel file, allow the user to choose a sheet
            if uploaded_file.name.endswith(('.xls', '.xlsx')):
                available_sheets = pd.ExcelFile(uploaded_file).sheet_names
                sheet_name = st.selectbox("Choose a sheet", available_sheets, index=0, key="sheet_selector")
            else:
                sheet_name = None

            st.write(f"Sheet: {sheet_name}")
            # Read the uploaded file into a DataFrame
            df_raw = read_file(uploaded_file, separator, sheet_name)

    else:  # "Link Input"
        # Allow the user to input a link to access the file
        link_input = st.text_input("Enter a link to access file:")
        # Read data from the provided link into a DataFrame
        df_raw = read_data_from_link(link_input, separator, sheet_name=None)

    # Check if the DataFrame is not None and not empty
    if df_raw is not None and not df_raw.empty:
        # Show the number of rows and columns
        st.write(f"Number of rows: {df_raw.shape[0]}, Number of columns: {df_raw.shape[1]}")

        # Display the raw data table
        # st.subheader("Raw Data:")
        # st.dataframe(df_raw, height=800)
        
        # Provide options for data processing operations
        st.subheader("Data Processing:")
        data_operations = ["describe","info", "head","tail",   "null_values", "drop_nulls", "unique_values", "value_counts","Sum","Maximum","Minimum"]
        data_operation = st.selectbox("Choose a data operation", data_operations)

        selected_column = None
        if data_operation in ["unique_values", "value_counts"]:
            selected_column = st.selectbox("Choose a column for operation:", df_raw.columns)
        # Perform the selected data processing operation
        if data_operation in ["unique_values", "value_counts"]:
            result = calculate_data(st, df_raw, data_operation, [selected_column])
        else:
            result = calculate_data(st, df_raw, data_operation)
        # Display the result if available
        if result is not None:
            st.write(f"{data_operation.capitalize()} Result:")
            st.table(result)
        # Operations Section
        st.subheader("Operations:")
        # List of basic operations
        operations = ["Median", "Mean", "Mode", "Variance", "Standard Deviation", "Range"]
        # Allow the user to choose an operation
        operation = st.selectbox("Choose an operation", operations, key="operation_selector")
        # Allow the user to choose multiple columns for the operation
        selected_columns = st.multiselect("Choose columns for operation:", df_raw.columns, key="columns_selector")
        # Perform the selected operation on the chosen columns
        result = calculate_operation(df_raw, operation, selected_columns)
        # Display the result or a warning if there's no result
        if result is not None:
            st.write(f"{operation.capitalize()} Value:")
            st.table(result)
        else:
            st.warning("No result to display.")
                
        # Statistics Operation Section
        st.subheader("Statistics Operation:")
        # List of statistics operations
        statistics_operations = [ "Skewness", "Kurtosis","Cumulative Product","Cumulative Sum",
                          "Absolute Change","Correlation","Covariance",
                          "IQR","Quantiles"]
        # Allow the user to choose a statistics operation
        statistics_operation = st.selectbox("Choose a statistics operation", statistics_operations, key="statistics_operation_selector")
        # Allow the user to choose multiple columns for the statistics operation
        selected_columns_stats = st.multiselect("Choose columns for statistics operation:", df_raw.columns, key="columns_stats_selector")
        # Perform the selected statistics operation on the chosen columns
        result_stats = calculate_statistics(df_raw, statistics_operation, selected_columns_stats)

        # Display the result if available
        if result_stats is not None:
            st.write(f"{statistics_operation.capitalize()} Result:")
            st.table(result_stats)
            
        # Indexation Section    
        indexation_section(df_raw)
        # Data Processing Section
        result = calculate_data(df_raw, operation, selected_columns)
        # Display the result if available
        if result is not None:
            st.write(f"{operation.capitalize()} Result:")
            st.table(result)
            
        # Visualization Section
        st.subheader("Visualization Section:")
        # Check if df_raw is not None before proceeding with visualization
        if df_raw is not None:  
            
            #Allow the user to choose a visualization type        
            user_selected_visualization = st.selectbox("Choose a visualization type:", list(visualization_types.keys()))

            # Allow the user to choose multiple columns for visualization
            user_selected_columns = st.multiselect("Choose columns for visualization:", df_raw.columns)
        
            st.subheader("Visualization:")
        
            # Perform the selected visualization operation on the chosen columns
            visualize_data(df_raw, user_selected_visualization, user_selected_columns)
            
        st.subheader("Statistical Tests:")
        # Allow the user to choose the type of statistical test
        statistical_test_type = st.selectbox("Choose a statistical test type:",
                                         ["t-test", "z-test", "chi-square test", "linear regression", "one-way-anova","two_way_anova"])
        if statistical_test_type == "t-test":
            # Allow the user to choose two columns for the t-test
            columns_for_t_test = st.multiselect("Choose two columns for t-test:", df_raw.columns)

            # Perform t-test if two columns are selected
            if len(columns_for_t_test) == 2:
                perform_t_test(st, df_raw, columns_for_t_test[0], columns_for_t_test[1])
            elif len(columns_for_t_test) > 2:
                st.warning("Please select only two columns for t-test.")

        elif statistical_test_type == "z-test":
            # Allow the user to choose columns for the z-test
            columns_for_z_test = st.multiselect("Choose columns for z-test:", df_raw.columns)

            # Check if at least one column is selected
            if not columns_for_z_test:
                st.warning("Please select at least one column for the z-test.")
            else:
                # Perform z-test based on the number of selected columns
                if len(columns_for_z_test) == 1:
                    # Perform one-sample z-test
                    perform_z_test(st, df_raw, columns_for_z_test[0])
                elif len(columns_for_z_test) == 2:
                    # Perform two-sample z-test
                    perform_z_test(st, df_raw, columns_for_z_test[0], columns_for_z_test[1])
                else:
                    st.warning("Please select one or two columns for the z-test.")

        elif statistical_test_type == "chi-square test":
            # Allow the user to choose two columns for the chi-square test
            columns_for_chi_square = st.multiselect("Choose two columns for chi-square test:", df_raw.columns)

            # Perform chi-square test if two columns are selected
            if len(columns_for_chi_square) == 2:
                perform_chi_square_test(st, df_raw, columns_for_chi_square[0], columns_for_chi_square[1])
            elif len(columns_for_chi_square) > 2:
                st.warning("Please select only two columns for chi-square test.")

        elif statistical_test_type == "linear regression":
            # Allow the user to choose two columns for linear regression
            columns_for_regression = st.multiselect("Choose two columns for linear regression:", df_raw.columns)

            # Perform linear regression if two columns are selected
            if len(columns_for_regression) == 2:
                perform_linear_regression(st, df_raw, columns_for_regression[0], columns_for_regression[1])
            elif len(columns_for_regression) > 2:
                st.warning("Please select only two columns for linear regression.")

        elif statistical_test_type == "one-way-anova":
            # Allow the user to choose columns for one-way ANOVA
            group_column = st.selectbox("Choose the group column for one-way ANOVA:", df_raw.columns)
            value_column = st.selectbox("Choose the value column for one-way ANOVA:", df_raw.columns)

            # Perform one-way ANOVA
            perform_one_way_anova(st, df_raw, group_column, value_column)

        elif statistical_test_type == "two_way_anova":
            # Allow the user to choose columns for two-way ANOVA
            group_column1 = st.selectbox("Choose the first group column for two-way ANOVA:", df_raw.columns)
            group_column2 = st.selectbox("Choose the second group column for two-way ANOVA:", df_raw.columns)
            value_column = st.selectbox("Choose the value column for two-way ANOVA:", df_raw.columns)

            # Perform two-way ANOVA
            try:
                two_way_anova(st, df_raw, group_column1, group_column2, value_column)
            except ValueError as e:
                st.error(f"Error performing two-way ANOVA: {str(e)}")

# Run the main function
if __name__ == "__main__":
    main()
