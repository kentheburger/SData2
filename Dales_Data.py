# Import required libraries
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
import glob  # to get the list of csv files
import os  # to extract filename from the path
import plotly.express as px
import base64

def fix_date_format(date_str):
    components = date_str.split("/")
    if len(components[2]) == 2:  # means it's in %d/%m/%y format
        components[2] = "20" + components[2]
    return "/".join(components)  # rearranging to %d/%m/%Y

# Function to load data from CSV files
@st.cache_data
def load_data(selected_month):
    required_columns = ['ItemSku', 'Post Code', 'DateSold', 'TotalSales', 'Price', 'TransactionNumber', 'CustomerAccount', 'Department', 'StoreName', 'ItemDescription']
    if selected_month == "Total":
        all_data = []
        for f in glob.glob("*.csv"):
            df = pd.read_csv(f)
            available_columns = [col for col in required_columns if col in df.columns]
            all_data.append(df[available_columns])
        return pd.concat(all_data, ignore_index=True)
    else:
        df = pd.read_csv(selected_month + ".csv")
        available_columns = [col for col in required_columns if col in df.columns]
        return df[available_columns]

# Add a selectbox in the sidebar for the months
selected_month = st.sidebar.selectbox("Select a month or time period", options=["Total"] + sorted([os.path.splitext(os.path.basename(x))[0] for x in glob.glob("*.csv")]))

# Load data based on the selected month
df = load_data(selected_month)

df['ItemSku'] = df['ItemSku'].astype(str)

# Handle missing values in the Post Code column
df['Post Code'] = df['Post Code'].fillna('Unknown')

# Fix date format and convert DateSold to datetime
df['DateSold'] = df['DateSold'].apply(fix_date_format)
df['DateSold'] = pd.to_datetime(df['DateSold'], format="%d/%m/%Y")

# Remove unwanted characters and convert to float
df['TotalSales'] = df['TotalSales'].apply(lambda x: float(x.replace('£', '').replace(',', '')))
df['Price'] = df['Price'].apply(lambda x: float(x.replace('£', '').replace(',', '')))

# Calculate profit per transaction
df['Profit'] = df['TotalSales'] - df['Price']

# Function to export user IDs to a CSV and provide a download link
def export_user_ids(df, item_description):
    user_ids = df[df['ItemDescription'] == item_description]['CustomerAccount'].unique()
    user_ids_df = pd.DataFrame(user_ids, columns=['CustomerAccount'])
    csv = user_ids_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{item_description}_user_ids.csv">Download {item_description} User IDs</a>'
    return href

# Displaying results in Streamlit
st.title("Scotsdales - For all the data in life")

# Weekly sales
st.subheader("Sales Over Time Selected")

# Group by week and sum the TotalSales
weekly_sales = df.resample('W', on='DateSold')['TotalSales'].sum()

# Group by week and count the unique CustomerAccounts as visits
weekly_visits = df.resample('W', on='DateSold')['CustomerAccount'].nunique()

# Convert both series into one DataFrame
data = pd.DataFrame({'Sales': weekly_sales, 'Visits': weekly_visits})

# Plot the line chart with both Sales and Visits
st.line_chart(data)

def calculate_total(column):
    return df[column].sum()

def calculate_unique(column):
    return df[column].nunique()

def calculate_percentage(condition, numerator_column, denominator_column):
    numerator = df[condition][numerator_column].sum()
    denominator = df[denominator_column].sum()
    return (numerator / denominator) * 100

def display_results(column, subheader, value):
    with column:
        st.subheader(subheader)
        st.markdown(f":red[`{value:.2f}`]")

# Total Spend 
total_spend = calculate_total('TotalSales')

# Total Visits
total_visits = calculate_unique('TransactionNumber')

# Spend divided by visits
average_spend_per_visit = total_spend / total_visits

# Create three columns
total_spend_col, total_visits_col, average_spend_col = st.columns(3)

display_results(total_spend_col, "Total Spend", total_spend)
display_results(total_visits_col, "Total Visits", total_visits)
display_results(average_spend_col, "Average Spend per Visit", average_spend_per_visit)

# Add a column to indicate if the transaction happened on a weekend
df['IsWeekend'] = df['DateSold'].dt.weekday >= 5

# % of Visits on Weekends
weekend_visits_pct = calculate_percentage(df['IsWeekend'], 'TransactionNumber', 'TransactionNumber')

# % of Spend on Weekends
weekend_spend_pct = calculate_percentage(df['IsWeekend'], 'TotalSales', 'TotalSales')

# Create two columns
weekend_visits_col, weekend_spend_col = st.columns(2)

display_results(weekend_visits_col, "% of Visits on Weekends", weekend_visits_pct)
display_results(weekend_spend_col, "% of Spend on Weekends", weekend_spend_pct)

# Sidebar controls
st.sidebar.subheader("Selections")

show_profitable_item = st.sidebar.checkbox("Show most profitable item")
show_postcode_groups = st.sidebar.checkbox("Show postcode groups")
show_popular_department = st.sidebar.checkbox("Show most popular department")
show_customer_department = st.sidebar.checkbox("Show customers by department")
show_customer_items = st.sidebar.checkbox("Show customers by item")
show_popular_store = st.sidebar.checkbox("Show most popular store")
show_profitable_store = st.sidebar.checkbox("Show most profitable store")
show_product_type_customers = st.sidebar.checkbox("Show customers by product type")
show_top_5_postcodes = st.sidebar.checkbox("Show top 5 postcodes")

# Sidebar controls for exporting top users
st.sidebar.subheader("Create Top User Export Per Department")
export_dept = st.sidebar.selectbox('Choose a department to export', df['Department'].unique())
export_button = st.sidebar.button('Export Top Users')

if export_button:
    # Filter the dataframe for the selected department
    export_filtered = df[df['Department'] == export_dept]

    # Group the filtered dataframe by CustomerAccount and sum the TotalSales, then get the top 100
    top_users = export_filtered.groupby('CustomerAccount')['TotalSales'].sum().nlargest(100).reset_index()

    # Create CSV file
    filename = f"{export_dept.replace(' ', '_')}_Top_Users.csv"
    top_users.to_csv(filename, index=False)

    st.sidebar.markdown(f'File {filename} has been created!')

# Main content

if show_profitable_item:
    most_profitable_item = df.groupby('ItemSku')['Profit'].sum().idxmax()
    st.header("Most profitable item")
    st.write(most_profitable_item)

if show_postcode_groups:
    postcode_groups = df.groupby('Post Code')['CustomerAccount'].unique()
    st.header("Postcode groups")
    st.write(postcode_groups)

if show_popular_department:
    popular_department = df.groupby('Department')['TotalSales'].sum().idxmax()
    st.header("Most popular department")
    st.write(popular_department)

if show_customer_department:
    customer_department = df.groupby('CustomerAccount')['Department'].unique()
    st.header("Customers by department")
    st.write(customer_department)

if show_customer_items:
    customer_items = df.groupby('CustomerAccount')['ItemSku'].unique()
    st.header("Customers by item")
    st.write(customer_items)

if show_popular_store:
    popular_store = df['StoreName'].value_counts().idxmax()
    st.header("Most popular store")
    st.write(popular_store)

if show_profitable_store:
    profitable_store = df.groupby('StoreName')['Profit'].sum().idxmax()
    st.header("Most profitable store")
    st.write(profitable_store)

if show_product_type_customers:
    product_type_customers = df.groupby('ItemDescription')['CustomerAccount'].unique()
    st.header("Customers by product type")
    st.write(product_type_customers)

if show_top_5_postcodes:
    st.header("Top 5 Postcodes")
    st.write(df['Post Code'].value_counts().head(5))

# Repurchase Rate
st.subheader("Repurchase Rate")

# Calculate the total transactions for each customer
total_transactions_per_customer = df.groupby('CustomerAccount')['TransactionNumber'].nunique()

# Calculate the number of customers who have shopped more than once
repeat_customers = total_transactions_per_customer[total_transactions_per_customer > 1].count()

# Calculate the repurchase rate
repurchase_rate = repeat_customers / df['CustomerAccount'].nunique()

# Display the repurchase rate
st.write(repurchase_rate)

st.subheader("Top Customers by Store and Department")

# Multi-select box for the store
selected_store = st.multiselect('Select a store', df['StoreName'].unique())

# Multi-select box for the department
selected_dept = st.multiselect('Select a department', df['Department'].unique())

for store in selected_store:
    for dept in selected_dept:
        # Filter the dataframe for the selected store and department
        df_filtered = df[(df['StoreName'] == store) & (df['Department'] == dept)]

        # Group the filtered dataframe by CustomerAccount and sum the TotalSales
        top_customers = df_filtered.groupby('CustomerAccount')['TotalSales'].sum().nlargest(50)

        st.write(f"Top customers for store {store} and department {dept}:")

        # Create a dropdown menu for customers
        selected_customer = st.selectbox('Choose a customer', top_customers.index.tolist())

        if selected_customer:
            customer_sales = top_customers[selected_customer]

            with st.expander(f'Customer {selected_customer} with Total Sales of {customer_sales} in department {dept}'):
                customer_data = df[df['CustomerAccount'] == selected_customer]
                
                # Purchase Value Analysis
                purchase_values = customer_data['TotalSales']
                st.write(f'Average Purchase Value: £{purchase_values.mean():.2f}')
                st.write(f'Maximum Purchase Value: £{purchase_values.max():.2f}')
                st.write(f'Minimum Purchase Value: £{purchase_values.min():.2f}')
                fig_purchase_values = px.histogram(purchase_values, nbins=50, labels={'value': 'Purchase Value', 'count': 'Frequency'})
                st.plotly_chart(fig_purchase_values)

                # Repeat Purchases Analysis
                repeat_purchases = customer_data['ItemDescription'].value_counts().nlargest(10)
                st.write(f'Top 10 Repeatedly Bought Items by {selected_customer}:')
                st.write(repeat_purchases)

                department_data = customer_data.groupby('Department')['TotalSales'].sum()

                # Diversity of Departments
                department_count = customer_data['Department'].nunique()
                total_departments = df['Department'].nunique()
                department_diversity_pct = (department_count / total_departments) * 100
                st.write(f'Diversity of departments: {department_count}/{total_departments} ({department_diversity_pct:.2f}%)')

                # Pie chart for department spending using Plotly
                fig = px.pie(department_data, values='TotalSales', names=department_data.index, title='Department Spending')
                st.plotly_chart(fig)

                # Pie chart for breakdown of visits by store location
                visits_data = customer_data['StoreName'].value_counts()
                fig_visits = px.pie(visits_data, values=visits_data.values, names=visits_data.index, title='Breakdown of Visits by Store Location')
                st.plotly_chart(fig_visits)

                # Make sure DateSold is in datetime format
                customer_data['DateSold'] = pd.to_datetime(customer_data['DateSold'])

                # Calculate the number of unique shopping days
                unique_days = customer_data['DateSold'].dt.date.nunique()

                # Calculate the percentage of purchases on weekdays vs weekends
                customer_data['Weekday'] = customer_data['DateSold'].dt.dayofweek
                weekdays = len(customer_data[(customer_data['Weekday'] >= 0) & (customer_data['Weekday'] <= 4)])
                weekends = len(customer_data[(customer_data['Weekday'] >= 5) & (customer_data['Weekday'] <= 6)])
                weekday_pct = weekdays / (weekdays + weekends) * 100
                weekend_pct = weekends / (weekdays + weekends) * 100

                st.write(f'Shopping data for customer {selected_customer} in other departments:')
                st.write(department_data)
                st.write(f'Number of unique shopping days: {unique_days}')
                st.write(f'Percentage of purchases on weekdays: {weekday_pct:.2f}%')
                st.write(f'Percentage of purchases on weekends: {weekend_pct:.2f}%')

                # Calculate sales per day for line chart
                sales_per_day = customer_data.groupby('DateSold')['TotalSales'].sum()
        
                # Generate line chart
                st.line_chart(sales_per_day)

# Function to get frequently bought together items
def get_frequently_bought_together(item_sku):
    transactions = df.groupby('TransactionNumber')['ItemSku'].apply(list)
    bought_together = defaultdict(set)

    for items in transactions:
        if item_sku in items:
            bought_together[item_sku].update([item for item in items if item != item_sku])

    # Let's just return the top 5 items frequently bought together with the specified SKU
    top_items = sorted(bought_together[item_sku], key=lambda x: transactions.apply(lambda items: x in items).sum(), reverse=True)[:20]
    # Get item descriptions
    top_items_descriptions = [df[df['ItemSku'] == sku]['ItemDescription'].values[0] for sku in top_items]

    return list(zip(top_items, top_items_descriptions))

st.subheader("Frequently Bought Together / Items to push selling together")
item_sku = st.text_input('Enter an item SKU')

if item_sku:
    frequently_bought_together = get_frequently_bought_together(item_sku)
    st.write(f"Items frequently bought with {item_sku}:")
    st.write(frequently_bought_together)

# Visit frequency per customer
st.subheader("Visit Frequency per Customer")

# Calculate visit frequency for each customer
visit_frequency = df.groupby('CustomerAccount')['TransactionNumber'].nunique().nlargest(50)

customer_account = st.text_input('Enter a customer account number')
if customer_account:
    st.write(f"Visit frequency for customer {customer_account}:")
    st.write(visit_frequency[customer_account])
else:
    st.write(visit_frequency)

# Finding the most frequently purchased item for a given customer
st.subheader("Predict next purchase for a customer")

# Text input for the customer account
customer_account = st.text_input("Enter a customer account")

# Check if the text input is not empty
if customer_account:
    try:
        # Convert the text input to an integer
        customer_account = int(customer_account)

        # Filter the dataframe for the given customer account
        df_customer = df[df['CustomerAccount'] == customer_account]

        # Find the most frequently purchased item
        next_purchase = df_customer['ItemSku'].mode()[0]
        next_purchase_description = df_customer[df['ItemSku'] == next_purchase]['ItemDescription'].unique()[0]

        st.write(f"The predicted next purchase for customer {customer_account} is {next_purchase} - {next_purchase_description}.")

    except ValueError:
        st.write("Please enter a valid customer account.")

st.subheader("Sales by Department")

# Calculate the total sales by department
department_sales = df.groupby('Department')['TotalSales'].sum().reset_index()

# Create a treemap
fig = px.treemap(department_sales, path=['Department'], values='TotalSales')

# Display the treemap in streamlit
st.plotly_chart(fig)

# Add a select box for the department
selected_dept = st.selectbox('Select a department to view item sales', df['Department'].unique())

if selected_dept:
    # Filter the dataframe for the selected department
    df_filtered = df[df['Department'] == selected_dept]
    
    # Display the department name
    st.header(f"Department: {selected_dept}")
    
    # Treemap option selection
    st.subheader("Treemap View Options")
    treemap_option = st.radio("Choose the basis for treemap visualization:", ('Total Sales (Money)', 'Volume (Quantity Sold)'))
    
    # Add slider for percentage selection
    st.subheader("Select Percentage Range for Top Products")
    percentage_range = st.slider("Select range", 0, 100, (0, 100))
    
    if treemap_option == 'Total Sales (Money)':
        # Group the filtered dataframe by ItemDescription and sum the TotalSales
        item_sales = df_filtered.groupby('ItemDescription')['TotalSales'].sum().reset_index()
        
        # Calculate the cumulative percentage
        item_sales['CumulativePercentage'] = item_sales['TotalSales'].cumsum() / item_sales['TotalSales'].sum() * 100
        
        # Filter based on the percentage range
        filtered_item_sales = item_sales[(item_sales['CumulativePercentage'] >= percentage_range[0]) & (item_sales['CumulativePercentage'] <= percentage_range[1])]
        
        # Create a treemap based on Total Sales
        fig = px.treemap(filtered_item_sales, path=['ItemDescription'], values='TotalSales', title='Total Sales (Money) by Product')
    
    else:
        # Group the filtered dataframe by ItemDescription and count the quantity sold
        item_sales = df_filtered.groupby('ItemDescription').size().reset_index(name='QuantitySold')
        
        # Calculate the cumulative percentage
        item_sales['CumulativePercentage'] = item_sales['QuantitySold'].cumsum() / item_sales['QuantitySold'].sum() * 100
        
        # Filter based on the percentage range
        filtered_item_sales = item_sales[(item_sales['CumulativePercentage'] >= percentage_range[0]) & (item_sales['CumulativePercentage'] <= percentage_range[1])]
        
        # Create a treemap based on Quantity Sold
        fig = px.treemap(filtered_item_sales, path=['ItemDescription'], values='QuantitySold', title='Volume (Quantity Sold) by Product')
    
    # Display the treemap in streamlit
    st.plotly_chart(fig)
    
    # Additional insights or patterns in product sales
    st.subheader(f"Top Products in {selected_dept} Department")
    top_products = df_filtered['ItemDescription'].value_counts().head(10)
    
    # Display top products with export buttons next to them
    for product in top_products.index:
        col1, col2 = st.columns([3, 1])
        col1.write(product)
        if col2.button(f"Export User IDs", key=product):
            href = export_user_ids(df_filtered, product)
            st.markdown(href, unsafe_allow_html=True)
    
    st.subheader(f"Purchase Patterns in {selected_dept} Department")
    purchase_patterns = df_filtered.groupby('ItemDescription')['TotalSales'].agg(['count', 'sum']).sort_values(by='sum', ascending=False)
    
    # Allow user to select how far they want to see the rankings
    num_products = st.number_input('How many products do you want to see in the rankings?', min_value=1, max_value=len(purchase_patterns), value=10)
    st.write(purchase_patterns.head(num_products))
    
    # Frequency of purchases and most common purchase day for each product
    st.subheader(f"Frequency of Purchases and Common Purchase Day in {selected_dept} Department")
    
    product_frequency = df_filtered['ItemDescription'].value_counts().reset_index()
    product_frequency.columns = ['ItemDescription', 'Frequency']
    
    # Calculate the most common day each product is purchased
    df_filtered['DayOfWeek'] = df_filtered['DateSold'].dt.day_name()
    most_common_day = df_filtered.groupby('ItemDescription')['DayOfWeek'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    most_common_day.columns = ['ItemDescription', 'MostCommonDay']
    
    # Merge the frequency and most common day dataframes
    product_stats = pd.merge(product_frequency, most_common_day, on='ItemDescription')
    st.write(product_stats)
    
    # Product analytics selection
    st.subheader("Select a Product for Detailed Analytics")
    selected_product = st.selectbox('Select a product', df_filtered['ItemDescription'].unique())
    
    if selected_product:
        product_data = df_filtered[df_filtered['ItemDescription'] == selected_product]
        
        # Product purchase trends over the week
        st.subheader(f"Purchase Trends Over the Week for {selected_product}")
        product_trends = product_data.groupby(['DayOfWeek']).size().reset_index(name='Counts')
        fig = px.line(product_trends, x='DayOfWeek', y='Counts', title=f'Purchase Trends Over the Week for {selected_product}')
        st.plotly_chart(fig)
        
        # Frequency of purchase
        product_data['DaysSinceLastPurchase'] = product_data['DateSold'].diff().dt.days
        avg_days_between_purchases = product_data['DaysSinceLastPurchase'].mean()
        st.write(f"This product is purchased every {avg_days_between_purchases:.2f} days on average.")
        
    # Search bar for products
    st.subheader("Search for a Product")
    
    product_search = st.text_input("Enter product name or SKU")
    
    if product_search:
        search_results = df_filtered[df_filtered['ItemDescription'].str.contains(product_search, case=False) | df_filtered['ItemSku'].str.contains(product_search)]
        
        if not search_results.empty:
            st.write(search_results[['ItemSku', 'ItemDescription', 'TotalSales', 'DateSold']])
            
            # Visualize purchase frequency and common purchase day for the searched product
            searched_product_stats = search_results['ItemDescription'].value_counts().reset_index()
            searched_product_stats.columns = ['ItemDescription', 'Frequency']
            searched_most_common_day = search_results.groupby('ItemDescription')['DayOfWeek'].agg(lambda x: x.value_counts().idxmax()).reset_index()
            searched_most_common_day.columns = ['ItemDescription', 'MostCommonDay']
            searched_product_stats = pd.merge(searched_product_stats, searched_most_common_day, on='ItemDescription')
            st.write(searched_product_stats)
            
            # Purchase trends for the searched product
            searched_product_trends = search_results.groupby(['DateSold']).size().reset_index(name='Counts')
            fig = px.line(searched_product_trends, x='DateSold', y='Counts', title=f'Sales Volume Over Time for {product_search}')
            st.plotly_chart(fig)
            
            # Ranking by purchase frequency for the searched product
            search_results['DaysSinceLastPurchase'] = search_results.groupby('ItemDescription')['DateSold'].diff().dt.days
            searched_product_ranking = search_results.groupby('ItemDescription')['DaysSinceLastPurchase'].mean().sort_values().reset_index()
            searched_product_ranking.columns = ['ItemDescription', 'AvgDaysBetweenPurchases']
            st.write(searched_product_ranking)
            for _, row in searched_product_ranking.iterrows():
                st.write(f"Product {row['ItemDescription']} is purchased every {row['AvgDaysBetweenPurchases']:.2f} days on average.")
            
        else:
            st.write("No products found.")
else:
    st.write("Please select a department to view the sales data.")

# Calculate the total profit by Post Code
postcode_profit = df.groupby('Post Code')['Profit'].sum().reset_index()

# Create a treemap using Plotly Express
fig_postcode = px.treemap(postcode_profit, path=['Post Code'], values='Profit', title='Profit by Post Code')

# Display the treemap in Streamlit
st.plotly_chart(fig_postcode)

# Monthly Spend
st.subheader("Monthly Spend")

# Calculate monthly spend
monthly_spend = df.resample('M', on='DateSold')['TotalSales'].sum()

# Make sure the index is a datetime index
monthly_spend.index = pd.to_datetime(monthly_spend.index)

st.line_chart(monthly_spend)

# Calculate the number of times each user has visited a store
st.subheader("User Store Visits")
store_visits = df.groupby(['CustomerAccount', 'StoreName'])['TransactionNumber'].nunique()
st.write(store_visits)

# Calculate the last visit date for each user
st.subheader("Last User Visit Date")
last_visit = df.groupby('CustomerAccount')['DateSold'].max()
st.write(last_visit)

# Add a select box for filtering users who haven't visited in more than 9 months
st.subheader("Inactive Users (No Visit in Last 9 Months)")

# Calculate the date 9 months ago from the latest date in the dataset
nine_months_ago = df['DateSold'].max() - pd.DateOffset(months=9)

# Get the last visit of each customer
last_visit = df.groupby('CustomerAccount')['DateSold'].max()

# Filter the last visit dataframe to get the inactive users
inactive_users = last_visit[last_visit < nine_months_ago]

# Calculate percentage of inactive users
inactive_users_percentage = (len(inactive_users) / len(last_visit)) * 100
active_users_percentage = 100 - inactive_users_percentage

# Display inactive users
st.write(inactive_users)

# Plotly Pie chart
fig = px.pie(values=[active_users_percentage, inactive_users_percentage], 
             names=['Active Users', 'Inactive Users'],
             title='User Activity')

st.plotly_chart(fig)

# Export inactive users list to CSV
if st.button('Export Inactive Users to CSV'):
    inactive_users.to_csv('inactive_users.csv')

# Estimate profit loss
average_profit_per_visit = df['Profit'].sum() / df['CustomerAccount'].nunique()
estimated_profit_loss = average_profit_per_visit * len(inactive_users)

st.subheader(f"Estimated Profit Loss from Inactive Users: £{estimated_profit_loss:.2f}")

# Add a subheader
st.subheader("Search for a Specific User")

# Create a text input for user search
search = st.text_input('Enter a user')

# If the search field is not empty, display information for the searched user
if search:
    if search in df['CustomerAccount'].values:
        st.write("Information for searched user:")

        # Filter dataframe for the specific user
        user_df = df[df['CustomerAccount'] == search]

        # Display total transaction value for the searched user
        total_transaction_value = user_df['TotalSales'].sum()
        st.write(f"Total transaction value: {total_transaction_value}")

        # Display average visit value for the searched user
        average_visit_value = user_df['TotalSales'].mean()
        st.write(f"Average visit value: {average_visit_value}")

        # Display the most common purchase SKU code for the searched user
        most_common_sku = user_df['ItemSku'].mode()[0]
        st.write(f"Most common purchase SKU: {most_common_sku}")

        # Display the most common department for the searched user
        most_common_department = user_df['Department'].mode()[0]
        st.write(f"Most common department: {most_common_department}")

        # Display total quantity sold to the searched user
        total_qty_sold = user_df['QtySold'].sum()
        st.write(f"Total quantity sold: {total_qty_sold}")

        # Display last purchase date for the searched user
        last_purchase_date = user_df['DateSold'].max()
        st.write(f"Last purchase date: {last_purchase_date}")

        # Check if the searched user is inactive
        if last_purchase_date < nine_months_ago:
            st.write(f"User {search} has been inactive for more than 9 months.")
        else:
            st.write(f"User {search} has visited the store in the last 9 months.")

# Top 500 loyalty customers by spend and number of purchases
st.subheader("Top 500 Loyalty Customers")

# Slider for selecting the number of top customers to view
num_top_customers = st.slider("Select the number of top customers to view", min_value=10, max_value=500, value=10)

# Top customers by spend
top_spenders = df.groupby('CustomerAccount')['TotalSales'].sum().nlargest(num_top_customers)
st.write(f"Top {num_top_customers} Customers by Spend")
st.write(top_spenders)

# Top customers by number of purchases
top_purchases = df.groupby('CustomerAccount')['TransactionNumber'].count().nlargest(num_top_customers)
st.write(f"Top {num_top_customers} Customers by Number of Purchases")
st.write(top_purchases)

# Function to categorize frequency of visits
def categorize_visits(df):
    df['DaysBetweenVisits'] = df.groupby('CustomerAccount')['DateSold'].diff().dt.days
    visit_frequency = df.groupby('CustomerAccount')['DaysBetweenVisits'].mean().dropna()
    
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 14, float('inf')]
    labels = ['Every day', 'Every 2 days', 'Every 3 days', 'Every 4 days', 'Every 5 days', 'Every 6 days', 'Once a week', 'Once every 2 weeks', 'More than 2 weeks']
    
    visit_categories = pd.cut(visit_frequency, bins=bins, labels=labels, right=False)
    return visit_categories

# Categorize visit frequency for top spenders
top_spenders_visits = categorize_visits(df[df['CustomerAccount'].isin(top_spenders.index)])
st.write("Visit Frequency Categories for Top Spenders")
st.write(top_spenders_visits.value_counts(normalize=True) * 100)

# Categorize visit frequency for top purchasers
top_purchases_visits = categorize_visits(df[df['CustomerAccount'].isin(top_purchases.index)])
st.write("Visit Frequency Categories for Top Purchasers")
st.write(top_purchases_visits.value_counts(normalize=True) * 100)

# Identify movers and shakers
st.subheader("Movers and Shakers")

def identify_movers_and_shakers(df, top_customers, num_customers):
    current_period = df['DateSold'].max()
    previous_period = current_period - pd.DateOffset(months=3)
    
    current_sales = df[df['DateSold'] > previous_period].groupby('CustomerAccount')['TotalSales'].sum()
    previous_sales = df[df['DateSold'] <= previous_period].groupby('CustomerAccount')['TotalSales'].sum()
    
    sales_change = (current_sales - previous_sales).dropna().sort_values()
    
    decreasing_activity = sales_change.head(num_customers)
    increasing_activity = sales_change.tail(num_customers)
    
    return decreasing_activity, increasing_activity

# Slider for selecting the number of movers and shakers to view
num_movers_and_shakers = st.slider("Select the number of movers and shakers to view", min_value=5, max_value=50, value=10)

decreasing_activity_spenders, increasing_activity_spenders = identify_movers_and_shakers(df, top_spenders, num_movers_and_shakers)
decreasing_activity_purchases, increasing_activity_purchases = identify_movers_and_shakers(df, top_purchases, num_movers_and_shakers)

st.write(f"Top {num_movers_and_shakers} Customers with Decreasing Activity (by Spend)")
st.write(decreasing_activity_spenders)

st.write(f"Top {num_movers_and_shakers} Customers with Increasing Activity (by Spend)")
st.write(increasing_activity_spenders)

st.write(f"Top {num_movers_and_shakers} Customers with Decreasing Activity (by Purchases)")
st.write(decreasing_activity_purchases)

st.write(f"Top {num_movers_and_shakers} Customers with Increasing Activity (by Purchases)")
st.write(increasing_activity_purchases)

# Line graph for movers and shakers
st.subheader("Movers and Shakers Line Graph")

def plot_movers_and_shakers(df, customer_ids, title):
    customer_sales = df[df['CustomerAccount'].isin(customer_ids)]
    customer_sales['Period'] = customer_sales['DateSold'].dt.to_period('M').astype(str)
    sales_over_time = customer_sales.groupby(['CustomerAccount', 'Period'])['TotalSales'].sum().reset_index()

    fig = px.line(sales_over_time, x='Period', y='TotalSales', color='CustomerAccount', title=title)
    st.plotly_chart(fig)

st.write("Top 50 Movers and Shakers by Spend")
plot_movers_and_shakers(df, increasing_activity_spenders.index.union(decreasing_activity_spenders.index), "Movers and Shakers by Spend")

st.write("Top 50 Movers and Shakers by Purchases")
plot_movers_and_shakers(df, increasing_activity_purchases.index.union(decreasing_activity_purchases.index), "Movers and Shakers by Purchases")

# Identify potential alternative garden centers based on post codes
st.subheader("Potential Alternative Garden Centers for Customers with Decreasing Activity")

# Coordinates for the given garden centers (using approximate coordinates for simplicity)
garden_centers = {
    'Blue Diamond': 'CB23 7PJ',
    'Dobbies Bury': 'IP33 2RN',
    'Dobbies Royston': 'SG8 6RB'
}

def extract_postcode_sector(postcode):
    return postcode.split()[0]

# Get the postcode sectors for the given garden centers
garden_center_postcode_sectors = {name: extract_postcode_sector(postcode) for name, postcode in garden_centers.items()}

def infer_alternative_garden_center(postcode):
    postcode_sector = extract_postcode_sector(postcode)
    for name, sector in garden_center_postcode_sectors.items():
        if sector == postcode_sector:
            return name
    return None

# Apply the inference function to the decreasing activity customers
decreasing_activity_customers = df[df['CustomerAccount'].isin(decreasing_activity_spenders.index)]
decreasing_activity_customers['AlternativeGardenCenter'] = decreasing_activity_customers['Post Code'].apply(infer_alternative_garden_center)

# Display the customers and their inferred alternative garden centers
alternative_centers = decreasing_activity_customers[['CustomerAccount', 'Post Code', 'AlternativeGardenCenter']].dropna()
st.write(alternative_centers)
