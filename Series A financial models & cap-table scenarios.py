

"""

# %pip install pandas plotly

"""## Revenue forecast module

### Subtask:
Create input fields for starting revenue, monthly growth, churn, and ARPU.

**Reasoning**:
Define and initialize the input variables for the revenue forecast module as instructed.
"""

# Revenue Forecast Inputs
starting_revenue = 10000  # Initial monthly revenue
monthly_growth_rate = 0.05  # 5% monthly revenue growth
churn_rate = 0.02  # 2% monthly churn rate
arpu = 50  # Average Revenue Per User

"""## Revenue forecast module

### Subtask:
Implement logic to calculate the 24-month revenue forecast.

**Reasoning**:
Implement the logic to calculate the 24-month revenue forecast based on the provided instructions.
"""



print("24-Month Revenue Forecast:")
print(revenue_forecast)

"""**Reasoning**:
The previous code block failed because the variables `starting_revenue`, `monthly_growth_rate`, `churn_rate`, and `arpu` were not defined in the current session. I need to redefine them before executing the forecast calculation.


"""



print("24-Month Revenue Forecast:")
print(revenue_forecast)

"""## Revenue forecast module

### Subtask:
Add scenario toggles for different growth assumptions.

**Reasoning**:
Define dictionaries for different scenarios and introduce a variable to select the scenario. Then update the revenue forecast calculation to use the selected scenario's parameters.
"""

# Define different scenario inputs
scenarios = {
    "Base Case": {
        "starting_revenue": 10000,
        "monthly_growth_rate": 0.05,
        "churn_rate": 0.02,
        "arpu": 50
    },
    "Optimistic": {
        "starting_revenue": 12000,
        "monthly_growth_rate": 0.07,
        "churn_rate": 0.01,
        "arpu": 55
    },
    "Pessimistic": {
        "starting_revenue": 8000,
        "monthly_growth_rate": 0.03,
        "churn_rate": 0.03,
        "arpu": 45
    }
}

# Select the scenario to use for the forecast
scenario = "Base Case"  # Change this to "Optimistic" or "Pessimistic" to see different forecasts

# Get parameters for the selected scenario
selected_scenario_params = scenarios[scenario]
starting_revenue = selected_scenario_params["starting_revenue"]
monthly_growth_rate = selected_scenario_params["monthly_growth_rate"]
churn_rate = selected_scenario_params["churn_rate"]
arpu = selected_scenario_params["arpu"]


# 1. Create a list to store the monthly revenue forecast
revenue_forecast = []

# 2. Initialize the first month's revenue
revenue_forecast.append(starting_revenue)

# Calculate starting users from starting revenue and ARPU
starting_users = starting_revenue / arpu
projected_users = starting_users

# 3. Iterate through the subsequent 23 months
for month in range(1, 24):
    # 4. Calculate the projected users for the current month
    projected_users = projected_users * (1 + monthly_growth_rate) * (1 - churn_rate)
    # 5. Calculate the revenue for the current month
    current_month_revenue = projected_users * arpu
    # 6. Append the calculated monthly revenue to the forecast list
    revenue_forecast.append(current_month_revenue)

print(f"24-Month Revenue Forecast ({scenario} Scenario):")
print(revenue_forecast)

"""## Revenue forecast module

### Subtask:
Generate a chart to visualize the revenue forecast.

**Reasoning**:
Generate a chart to visualize the revenue forecast using Plotly.
"""

import plotly.graph_objects as go

# 2. Create a list of month numbers from 1 to 24 to represent the x-axis.
months = list(range(1, 25))

# 3. Create a Plotly figure using go.Figure().
fig = go.Figure()

# 4. Add a scatter plot trace to the figure using go.Scatter. Use the list of month numbers for the x-axis and the revenue_forecast list for the y-axis. Name the trace appropriately (e.g., the selected scenario name).
fig.add_trace(go.Scatter(x=months, y=revenue_forecast, mode='lines+markers', name=scenario))

# 5. Update the layout of the figure to add a title (e.g., "24-Month Revenue Forecast"), x-axis label ("Month"), and y-axis label ("Revenue").
fig.update_layout(
    title="24-Month Revenue Forecast",
    xaxis_title="Month",
    yaxis_title="Revenue ($)"
)

# 6. Display the plot using the .show() method of the figure.
fig.show()

"""## Unit economics stress test module

### Subtask:
Create input fields for CAC, LTV, and fixed/variable cost per unit.

**Reasoning**:
Define variables for CAC, LTV, fixed cost per unit, and variable cost per unit.
"""

# Unit Economics Stress Test Inputs
cac = 100  # Customer Acquisition Cost
ltv = 500  # Customer Lifetime Value
fixed_cost_per_unit = 10  # Fixed cost associated with each unit
variable_cost_per_unit = 20  # Variable cost associated with each unit

"""## Unit economics stress test module

### Subtask:
Implement logic to calculate LTV/CAC ratio and contribution margin.

**Reasoning**:
Calculate the LTV to CAC ratio and the contribution margin per unit based on the provided input variables and print the results.
"""

# Calculate LTV to CAC ratio
ltv_cac_ratio = ltv / cac

# Calculate contribution margin per unit
contribution_margin_per_unit = arpu - (fixed_cost_per_unit + variable_cost_per_unit)

# Print the calculated values
print(f"LTV/CAC Ratio: {ltv_cac_ratio}")
print(f"Contribution Margin Per Unit: {contribution_margin_per_unit}")

"""## Unit economics stress test module

### Subtask:
Create a stress grid to show how changes in inputs affect key metrics.

**Reasoning**:
Import numpy to create ranges for CAC and LTV, then create the stress grid DataFrame by iterating through the ranges and calculating the LTV/CAC ratio and contribution margin for each combination. Finally, display the resulting DataFrame.
"""

import numpy as np

# Define ranges for CAC and LTV
cac_range = np.linspace(50, 200, 10) # Example: CAC from 50 to 200 in 10 steps
ltv_range = np.linspace(200, 800, 10) # Example: LTV from 200 to 800 in 10 steps

# Create a list to store the results
stress_test_results = []

# Iterate through all combinations of CAC and LTV
for current_cac in cac_range:
    for current_ltv in ltv_range:
        # Calculate LTV/CAC ratio
        current_ltv_cac_ratio = current_ltv / current_cac

        # Calculate contribution margin per unit (using existing variables)
        # contribution_margin_per_unit = arpu - (fixed_cost_per_unit + variable_cost_per_unit)
        # The contribution margin per unit is not dependent on CAC or LTV, so it will be constant across the grid based on the previously defined inputs.

        # Store the results in a dictionary
        result = {
            'CAC': current_cac,
            'LTV': current_ltv,
            'LTV/CAC Ratio': current_ltv_cac_ratio,
            'Contribution Margin Per Unit': contribution_margin_per_unit # Using the previously calculated value
        }

        # Append the dictionary to the results list
        stress_test_results.append(result)

# Convert the list of results into a pandas DataFrame
stress_grid_df = pd.DataFrame(stress_test_results)

# Display the stress grid DataFrame
display(stress_grid_df)

"""**Reasoning**:
The previous code failed because pandas was not imported. I need to import pandas and then regenerate the stress grid DataFrame and display it.


"""

import pandas as pd
import numpy as np

# Define ranges for CAC and LTV
cac_range = np.linspace(50, 200, 10) # Example: CAC from 50 to 200 in 10 steps
ltv_range = np.linspace(200, 800, 10) # Example: LTV from 200 to 800 in 10 steps

# Create a list to store the results
stress_test_results = []

# Iterate through all combinations of CAC and LTV
for current_cac in cac_range:
    for current_ltv in ltv_range:
        # Calculate LTV/CAC ratio
        current_ltv_cac_ratio = current_ltv / current_cac

        # Calculate contribution margin per unit (using existing variables)
        # contribution_margin_per_unit = arpu - (fixed_cost_per_unit + variable_cost_per_unit)
        # The contribution margin per unit is not dependent on CAC or LTV, so it will be constant across the grid based on the previously defined inputs.

        # Store the results in a dictionary
        result = {
            'CAC': current_cac,
            'LTV': current_ltv,
            'LTV/CAC Ratio': current_ltv_cac_ratio,
            'Contribution Margin Per Unit': contribution_margin_per_unit # Using the previously calculated value
        }

        # Append the dictionary to the results list
        stress_test_results.append(result)

# Convert the list of results into a pandas DataFrame
stress_grid_df = pd.DataFrame(stress_test_results)

# Display the stress grid DataFrame
display(stress_grid_df)

"""## Cap table simulator module

### Subtask:
Create input fields for pre-money valuation, round size, current cap table details (founders, pre-seed, ESOP), and post-round ESOP percentage.

**Reasoning**:
Define the input variables for the Cap Table Simulator module as requested by the subtask.
"""

# Cap Table Simulator Inputs
pre_money_valuation = 10_000_000  # Example pre-money valuation in USD
round_size = 2_000_000  # Example funding round size in USD

# Current Cap Table (represented as a dictionary of equity ownership percentages)
current_cap_table = {
    'Founders': 0.70,       # 70% ownership
    'Pre-Seed Investors': 0.20, # 20% ownership
    'ESOP': 0.10            # 10% ownership (existing pool)
}

# Desired post-round ESOP percentage
post_round_esop_percentage = 0.15 # 15% of the company's equity after the round

"""## Cap table simulator module

### Subtask:
Implement logic to calculate the post-money cap table, share price, and dilution breakdown.

**Reasoning**:
Implement the logic to calculate the post-money cap table, share price, and dilution breakdown according to the instructions. This involves calculating the post-money valuation, determining the pre-money and post-money share counts, calculating the post-money share price, determining the new shares issued for the round and the ESOP, and finally calculating the post-money ownership percentages and dilution for each group.
"""

# 1. Calculate the post-money valuation
post_money_valuation = pre_money_valuation + round_size
print(f"Post-Money Valuation: ${post_money_valuation:,.2f}")

# 2. Assume a simple starting value like $1 per share for calculation purposes to derive initial share count
# We need a pre-money share count to calculate the post-money share price and new shares.
# Let's assume an arbitrary initial price per share to derive the pre-money share count.
# A common approach is to assume a nominal price, e.g., $1, or derive from a known share count.
# Since we only have valuation and percentages, let's work with a notional share count.
# Let's assume a total pre-money shares that corresponds to the pre-money valuation
# at a notional price per share. We can set the notional pre-money price per share.
# Let's assume a notional pre-money share price of $1.
notional_pre_money_share_price = 1.0
total_pre_money_shares = pre_money_valuation / notional_pre_money_share_price
print(f"Total Pre-Money Shares (notional): {total_pre_money_shares:,.0f}")

# 6. Calculate the number of shares held by each existing shareholder group before the round
pre_money_shares_distribution = {}
for group, percentage in current_cap_table.items():
    pre_money_shares_distribution[group] = total_pre_money_shares * percentage
    print(f"Pre-money shares for {group}: {pre_money_shares_distribution[group]:,.0f}")

# Now we need to calculate the post-money share price and new shares.
# The post-money share price is Post-Money Valuation / Total Post-Money Shares.
# Total Post-Money Shares = Total Pre-Money Shares + New Round Shares + New ESOP Shares.
# New Round Shares = Round Size / Post-Money Share Price.
# Target Post-Money ESOP Shares = Post-Money Valuation * Post-Round ESOP Percentage / Post-Money Share Price
# New ESOP Shares = Target Post-Money ESOP Shares - Existing ESOP Shares

# This forms a system of equations. Let S_post be the total post-money shares and P_post be the post-money share price.
# P_post = Post_Money_Valuation / S_post
# S_post = Total_Pre_Money_Shares + (Round_Size / P_post) + New_ESOP_Shares
# New_ESOP_Shares = (Post_Round_ESOP_Percentage * S_post) - Existing_ESOP_Shares
# Substitute P_post into the second equation:
# S_post = Total_Pre_Money_Shares + (Round_Size / (Post_Money_Valuation / S_post)) + New_ESOP_Shares
# S_post = Total_Pre_Money_Shares + (Round_Size * S_post / Post_Money_Valuation) + New_ESOP_Shares
# S_post * (1 - Round_Size / Post_Money_Valuation) = Total_Pre_Money_Shares + New_ESOP_Shares
# S_post * (Post_Money_Valuation - Round_Size) / Post_Money_Valuation = Total_Pre_Money_Shares + New_ESOP_Shares
# S_post * Pre_Money_Valuation / Post_Money_Valuation = Total_Pre_Money_Shares + New_ESOP_Shares

# Now substitute New_ESOP_Shares:
# S_post * Pre_Money_Valuation / Post_Money_Valuation = Total_Pre_Money_Shares + (Post_Round_ESOP_Percentage * S_post) - Existing_ESOP_Shares
# S_post * Pre_Money_Valuation / Post_Money_Valuation - (Post_Round_ESOP_Percentage * S_post) = Total_Pre_Money_Shares - Existing_ESOP_Shares
# S_post * (Pre_Money_Valuation / Post_Money_Valuation - Post_Round_ESOP_Percentage) = Total_Pre_Money_Shares - Existing_ESOP_Shares

# S_post = (Total_Pre_Money_Shares - Existing_ESOP_Shares) / (Pre_Money_Valuation / Post_Money_Valuation - Post_Round_ESOP_Percentage)
# Existing_ESOP_Shares = pre_money_shares_distribution['ESOP']

# Calculate total post-money shares based on the formula derived
# Ensure the denominator is not zero or negative
denominator = (pre_money_valuation / post_money_valuation) - post_round_esop_percentage
if denominator <= 0:
    print("Error: Cannot calculate post-money shares with the given inputs. Check valuation and ESOP percentage.")
    # Set values to None or handle error appropriately
    total_post_money_shares = None
    post_money_share_price = None
    new_round_shares = None
    new_esop_shares = None
else:
    total_post_money_shares = (total_pre_money_shares - pre_money_shares_distribution['ESOP']) / denominator
    print(f"Total Post-Money Shares (calculated): {total_post_money_shares:,.0f}")

    # 4. Calculate the post-money share price
    post_money_share_price = post_money_valuation / total_post_money_shares
    print(f"Post-Money Share Price: ${post_money_share_price:,.4f}")

    # 3. Calculate the total number of new shares issued in the round
    new_round_shares = round_size / post_money_share_price
    print(f"New Shares Issued in Round: {new_round_shares:,.0f}")

    # 7. Calculate the number of new shares for the ESOP pool
    target_post_money_esop_shares = post_round_esop_percentage * total_post_money_shares
    new_esop_shares = target_post_money_esop_shares - pre_money_shares_distribution['ESOP']
    # Ensure new_esop_shares is not negative (shouldn't be if calculation is correct, but good check)
    new_esop_shares = max(0, new_esop_shares)
    print(f"Target Post-Money ESOP Shares: {target_post_money_esop_shares:,.0f}")
    print(f"New ESOP Shares Added: {new_esop_shares:,.0f}")

    # 5. Verify total number of shares after the round (existing + new round + new ESOP)
    verified_total_post_money_shares = total_pre_money_shares + new_round_shares + new_esop_shares
    print(f"Verified Total Post-Money Shares: {verified_total_post_money_shares:,.0f}")
    # This should be very close to total_post_money_shares calculated earlier

    # 8. Update the shares for each existing shareholder group and the ESOP after the new shares
    post_money_shares_distribution = pre_money_shares_distribution.copy()
    post_money_shares_distribution['ESOP'] += new_esop_shares
    post_money_shares_distribution['New Investors'] = new_round_shares

    print("\nPost-Money Share Distribution:")
    for group, shares in post_money_shares_distribution.items():
        print(f"{group}: {shares:,.0f} shares")


    # 9. Calculate the ownership percentage for each group in the post-money cap table
    post_money_cap_table = {}
    for group, shares in post_money_shares_distribution.items():
        post_money_cap_table[group] = shares / total_post_money_shares
        print(f"Post-money ownership for {group}: {post_money_cap_table[group]:.2%}")

    # 10. Create a dictionary or similar structure to represent the post-money cap table
    # We already have post_money_cap_table dictionary

    # 11. Calculate the dilution for each existing shareholder group
    dilution_breakdown = {}
    for group, pre_money_percentage in current_cap_table.items():
        if group in post_money_cap_table:
            post_money_percentage = post_money_cap_table[group]
            dilution = pre_money_percentage - post_money_percentage
            dilution_breakdown[group] = dilution
            print(f"Dilution for {group}: {dilution:.2%}")
        else:
            # This case should not happen if the group exists in current_cap_table
            dilution_breakdown[group] = pre_money_percentage # They were completely diluted (unlikely for core groups)

    print("\nDilution Breakdown:")
    for group, dilution in dilution_breakdown.items():
         print(f"{group}: {dilution:.2%}")

"""## Cap table simulator module

### Subtask:
Generate a chart to visualize the dilution breakdown.

**Reasoning**:
Generate a bar chart to visualize the dilution breakdown using Plotly.
"""

import plotly.graph_objects as go

# 2. Create a list of shareholder groups from the dilution_breakdown dictionary keys.
shareholder_groups = list(dilution_breakdown.keys())

# 3. Create a list of dilution percentages from the dilution_breakdown dictionary values.
dilution_percentages = list(dilution_breakdown.values())

# 4. Create a Plotly bar chart using go.Figure() and go.Bar(). Use the list of shareholder groups for the x-axis and the list of dilution percentages for the y-axis.
fig = go.Figure(data=[go.Bar(x=shareholder_groups, y=dilution_percentages)])

# 5. Update the layout of the figure to add a title (e.g., "Dilution Breakdown by Shareholder Group"), x-axis label ("Shareholder Group"), and y-axis label ("Dilution (%)"). Format the y-axis as percentages.
fig.update_layout(
    title="Dilution Breakdown by Shareholder Group",
    xaxis_title="Shareholder Group",
    yaxis_title="Dilution (%)",
    yaxis_tickformat=".1%" # Format y-axis as percentage with one decimal place
)

# 6. Display the plot using the .show() method.
fig.show()

"""## Term sheet comparison module

### Subtask:
Create input fields for 2-3 offer scenarios (valuation and round size).

**Reasoning**:
Define a dictionary to store multiple offer scenarios with their respective valuations and round sizes.
"""

# Term Sheet Comparison Inputs
offer_scenarios = {
    'Offer A': {
        'valuation': 12_000_000,
        'round_size': 2_000_000
    },
    'Offer B': {
        'valuation': 15_000_000,
        'round_size': 3_000_000
    }
    # Add more scenarios as needed
}

print("Defined Offer Scenarios:")
for offer, details in offer_scenarios.items():
    print(f"- {offer}: Valuation=${details['valuation']:,.0f}, Round Size=${details['round_size']:,.0f}")

"""## Term sheet comparison module

### Subtask:
Implement logic to calculate investor percentage, founder percentage, and post-money valuation for each scenario.

**Reasoning**:
Implement the logic to calculate investor percentage, founder percentage, and post-money valuation for each scenario by iterating through the offer_scenarios dictionary, calculating the required metrics, and storing them. Finally, print the results.
"""

# List to store the calculated metrics for each scenario
scenario_metrics = []

# Iterate through each offer scenario
for offer_name, details in offer_scenarios.items():
    valuation = details['valuation']
    round_size = details['round_size']

    # Calculate post-money valuation
    post_money_valuation = valuation + round_size

    # Calculate investor percentage (New Investors)
    investor_percentage = round_size / post_money_valuation

    # Assuming 'Founders' percentage is part of the existing ownership that gets diluted
    # For simplicity in this comparison module, let's calculate the founder percentage
    # relative to the pre-money ownership structure if the round happens.
    # However, the prompt asks for "founder %" in the output for the *Term Sheet Comparison*
    # module, which usually refers to their ownership *after* the round under that specific term sheet.
    # This requires knowing the *current* founder percentage before the round.
    # Let's use the 'Founders' percentage from the previously defined `current_cap_table`.
    # The founder percentage *after* the round would be their pre-money percentage * (Pre-Money Valuation / Post-Money Valuation)
    # assuming no new shares are issued to founders and the ESOP is handled separately.
    # However, the prompt asks for 'founder %' in the context of the offer itself,
    # which is a bit ambiguous. Let's interpret 'founder %' here as their percentage
    # of the company *after* the round, based on their initial percentage being diluted
    # by the new money and potentially a new ESOP pool (though the ESOP calculation
    # was done in the previous module and isn't explicitly tied to each offer here).
    # A simpler interpretation for a *Term Sheet Comparison* might be the percentage
    # the *new investors* get vs. the percentage *retained by existing shareholders*,
    # where founders are a part of the existing shareholders.
    # Let's calculate the percentage retained by *all* existing shareholders,
    # which is (Pre-Money Valuation / Post-Money Valuation). The founder percentage
    # *within* the existing group remains the same relative to other existing shareholders,
    # but their overall percentage of the company decreases.
    # Let's calculate the percentage retained by existing shareholders.
    existing_shareholder_percentage_retained = valuation / post_money_valuation

    # We can also show the founders' specific percentage if we use the current_cap_table['Founders']
    # founder_percentage_post_round = current_cap_table['Founders'] * existing_shareholder_percentage_retained

    # Let's focus on the split between New Investors and Existing Shareholders as this is common in term sheet summaries.
    # The prompt specifically asks for "founder %" and "investor %".
    # Let's assume "investor %" means the new investors, and "founder %" means the founders' percentage *post-money*.
    # This requires using the current_cap_table['Founders'] from the previous module.
    founder_percentage_pre_money = current_cap_table['Founders']
    founder_percentage_post_money = founder_percentage_pre_money * (valuation / post_money_valuation)


    # Store the metrics for the current scenario
    scenario_metrics.append({
        'Offer': offer_name,
        'Pre-Money Valuation': valuation,
        'Round Size': round_size,
        'Post-Money Valuation': post_money_valuation,
        'New Investor Percentage': investor_percentage,
        'Founder Percentage (Post-Money)': founder_percentage_post_money # Using calculated post-money founder %
    })

# Print the calculated metrics for each offer scenario
print("\nTerm Sheet Comparison Metrics:")
for metrics in scenario_metrics:
    print(f"\n{metrics['Offer']}:")
    print(f"  Pre-Money Valuation: ${metrics['Pre-Money Valuation']:,.0f}")
    print(f"  Round Size: ${metrics['Round Size']:,.0f}")
    print(f"  Post-Money Valuation: ${metrics['Post-Money Valuation']:,.0f}")
    print(f"  New Investor Percentage: {metrics['New Investor Percentage']:.2%}")
    print(f"  Founder Percentage (Post-Money): {metrics['Founder Percentage (Post-Money)']:.2%}")

"""## Term sheet comparison module

### Subtask:
Generate a side-by-side bar chart to compare the scenarios.

**Reasoning**:
Import the necessary module for plotting and create the lists required for the chart.
"""

import plotly.graph_objects as go

# Create a list of offer names
offer_names = [scenario['Offer'] for scenario in scenario_metrics]

# Create lists for metrics
post_money_valuations = [scenario['Post-Money Valuation'] for scenario in scenario_metrics]
new_investor_percentages = [scenario['New Investor Percentage'] for scenario in scenario_metrics]
founder_percentages = [scenario['Founder Percentage (Post-Money)'] for scenario in scenario_metrics]

"""**Reasoning**:
Create the Plotly figure and add bar traces for each metric to generate a grouped bar chart.


"""

# Create a Plotly figure
fig = go.Figure()

# Add bar chart traces for each metric
fig.add_trace(go.Bar(
    x=offer_names,
    y=post_money_valuations,
    name='Post-Money Valuation',
    yaxis='y1' # Assign to the primary y-axis
))

fig.add_trace(go.Bar(
    x=offer_names,
    y=new_investor_percentages,
    name='New Investor Percentage',
    yaxis='y2' # Assign to the secondary y-axis
))

fig.add_trace(go.Bar(
    x=offer_names,
    y=founder_percentages,
    name='Founder Percentage (Post-Money)',
    yaxis='y2' # Assign to the secondary y-axis
))

# Update the layout for a grouped bar chart with two y-axes
fig.update_layout(
    title="Term Sheet Comparison",
    xaxis_title="Offer Scenario",
    yaxis=dict(
        title="Post-Money Valuation ($)",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="Percentage",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right",
        tickformat=".1%" # Format as percentage
    ),
    barmode='group', # Group bars for each offer
    legend=dict(x=0, y=1.1, orientation="h") # Position legend
)

# Display the plot
fig.show()

"""## Integrate modules and add ui (optional)

### Subtask:
Combine the modules into a single notebook.

**Reasoning**:
Create a new code cell and copy and paste the code from all previous completed subtasks in the correct order, adding markdown cells for clarity and ensuring all imports are at the beginning. Then, run the combined cell to verify its execution.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

## Revenue Forecast Module

# Revenue Forecast Inputs
scenarios = {
    "Base Case": {
        "starting_revenue": 10000,
        "monthly_growth_rate": 0.05,
        "churn_rate": 0.02,
        "arpu": 50
    },
    "Optimistic": {
        "starting_revenue": 12000,
        "monthly_growth_rate": 0.07,
        "churn_rate": 0.01,
        "arpu": 55
    },
    "Pessimistic": {
        "starting_revenue": 8000,
        "monthly_growth_rate": 0.03,
        "churn_rate": 0.03,
        "arpu": 45
    }
}

# Select the scenario to use for the forecast
scenario = "Base Case"  # Change this to "Optimistic" or "Pessimistic" to see different forecasts

# Get parameters for the selected scenario
selected_scenario_params = scenarios[scenario]
starting_revenue = selected_scenario_params["starting_revenue"]
monthly_growth_rate = selected_scenario_params["monthly_growth_rate"]
churn_rate = selected_scenario_params["churn_rate"]
arpu = selected_scenario_params["arpu"]

# 1. Create a list to store the monthly revenue forecast
revenue_forecast = []

# 2. Initialize the first month's revenue
revenue_forecast.append(starting_revenue)

# Calculate starting users from starting revenue and ARPU
starting_users = starting_revenue / arpu
projected_users = starting_users

# 3. Iterate through the subsequent 23 months
for month in range(1, 24):
    # 4. Calculate the projected users for the current month
    projected_users = projected_users * (1 + monthly_growth_rate) * (1 - churn_rate)
    # 5. Calculate the revenue for the current month
    current_month_revenue = projected_users * arpu
    # 6. Append the calculated monthly revenue to the forecast list
    revenue_forecast.append(current_month_revenue)

print(f"24-Month Revenue Forecast ({scenario} Scenario):")
# print(revenue_forecast) # Printing the full list can be long, let's rely on the chart

# Generate Revenue Forecast Chart
# 2. Create a list of month numbers from 1 to 24 to represent the x-axis.
months = list(range(1, 25))

# 3. Create a Plotly figure using go.Figure().
fig = go.Figure()

# 4. Add a scatter plot trace to the figure using go.Scatter. Use the list of month numbers for the x-axis and the revenue_forecast list for the y-axis. Name the trace appropriately (e.g., the selected scenario name).
fig.add_trace(go.Scatter(x=months, y=revenue_forecast, mode='lines+markers', name=scenario))

# 5. Update the layout of the figure to add a title (e.g., "24-Month Revenue Forecast"), x-axis label ("Month"), and y-axis label ("Revenue").
fig.update_layout(
    title="24-Month Revenue Forecast",
    xaxis_title="Month",
    yaxis_title="Revenue ($)"
)

# 6. Display the plot using the .show() method of the figure.
fig.show()


## Unit Economics Stress Test Module

# Unit Economics Stress Test Inputs
cac = 100  # Customer Acquisition Cost
ltv = 500  # Customer Lifetime Value
fixed_cost_per_unit = 10  # Fixed cost associated with each unit
variable_cost_per_unit = 20  # Variable cost associated with each unit

# Calculate LTV to CAC ratio
ltv_cac_ratio = ltv / cac

# Calculate contribution margin per unit
contribution_margin_per_unit = arpu - (fixed_cost_per_unit + variable_cost_per_unit)

# Print the calculated values
print(f"\nUnit Economics Metrics:")
print(f"LTV/CAC Ratio: {ltv_cac_ratio}")
print(f"Contribution Margin Per Unit: {contribution_margin_per_unit}")

# Create Stress Grid
# Define ranges for CAC and LTV
cac_range = np.linspace(50, 200, 10) # Example: CAC from 50 to 200 in 10 steps
ltv_range = np.linspace(200, 800, 10) # Example: LTV from 200 to 800 in 10 steps

# Create a list to store the results
stress_test_results = []

# Iterate through all combinations of CAC and LTV
for current_cac in cac_range:
    for current_ltv in ltv_range:
        # Calculate LTV/CAC ratio
        current_ltv_cac_ratio = current_ltv / current_cac

        # Store the results in a dictionary
        result = {
            'CAC': current_cac,
            'LTV': current_ltv,
            'LTV/CAC Ratio': current_ltv_cac_ratio,
            'Contribution Margin Per Unit': contribution_margin_per_unit # Using the previously calculated value
        }

        # Append the dictionary to the results list
        stress_test_results.append(result)

# Convert the list of results into a pandas DataFrame
stress_grid_df = pd.DataFrame(stress_test_results)

# Display the stress grid DataFrame
print("\nUnit Economics Stress Grid (LTV/CAC Ratio and Contribution Margin):")
display(stress_grid_df)


## Cap Table Simulator Module

# Cap Table Simulator Inputs
pre_money_valuation = 10_000_000  # Example pre-money valuation in USD
round_size = 2_000_000  # Example funding round size in USD

# Current Cap Table (represented as a dictionary of equity ownership percentages)
current_cap_table = {
    'Founders': 0.70,       # 70% ownership
    'Pre-Seed Investors': 0.20, # 20% ownership
    'ESOP': 0.10            # 10% ownership (existing pool)
}

# Desired post-round ESOP percentage
post_round_esop_percentage = 0.15 # 15% of the company's equity after the round

# Calculate Post-Money Cap Table, Share Price, and Dilution
# 1. Calculate the post-money valuation
post_money_valuation = pre_money_valuation + round_size
print(f"\nCap Table Simulation:")
print(f"Post-Money Valuation: ${post_money_valuation:,.2f}")

# 2. Assume a simple starting value like $1 per share for calculation purposes to derive initial share count
# We need a pre-money share count to calculate the post-money share price and new shares.
# Let's assume an arbitrary initial price per share to derive the pre-money share count.
# A common approach is to assume a nominal price, e.g., $1, or derive from a known share count.
# Since we only have valuation and percentages, let's work with a notional share count.
# Let's assume a total pre-money shares that corresponds to the pre-money valuation
# at a notional price per share. We can set the notional pre-money price per share.
# Let's assume a notional pre-money share price of $1.
notional_pre_money_share_price = 1.0
total_pre_money_shares = pre_money_valuation / notional_pre_money_share_price
print(f"Total Pre-Money Shares (notional): {total_pre_money_shares:,.0f}")

# 6. Calculate the number of shares held by each existing shareholder group before the round
pre_money_shares_distribution = {}
for group, percentage in current_cap_table.items():
    pre_money_shares_distribution[group] = total_pre_money_shares * percentage
    print(f"Pre-money shares for {group}: {pre_money_shares_distribution[group]:,.0f}")

# Now we need to calculate the post-money share price and new shares.
# The post-money share price is Post-Money Valuation / Total Post-Money Shares.
# Total Post-Money Shares = Total Pre-Money Shares + New Round Shares + New ESOP Shares.
# New Round Shares = Round Size / Post-Money Share Price.
# Target Post-Money ESOP Shares = Post-Money Valuation * Post-Round ESOP Percentage / Post-Money Share Price
# New ESOP Shares = Target Post-Money ESOP Shares - Existing ESOP Shares

# This forms a system of equations. Let S_post be the total post-money shares and P_post be the post-money share price.
# P_post = Post_Money_Valuation / S_post
# S_post = Total_Pre_Money_Shares + (Round_Size / P_post) + New_ESOP_Shares
# New_ESOP_Shares = (Post_Round_ESOP_Percentage * S_post) - Existing_ESOP_Shares
# Substitute P_post into the second equation:
# S_post = Total_Pre_Money_Shares + (Round_Size / (Post_Money_Valuation / S_post)) + New_ESOP_Shares
# S_post = Total_Pre_Money_Shares + (Round_Size * S_post / Post_Money_Valuation) + New_ESOP_Shares
# S_post * (1 - Round_Size / Post_Money_Valuation) = Total_Pre_Money_Shares + New_ESOP_Shares
# S_post * (Post_Money_Valuation - Round_Size) / Post_Money_Valuation = Total_Pre_Money_Shares + New_ESOP_Shares
# S_post * Pre_Money_Valuation / Post_Money_Valuation = Total_Pre_Money_Shares + New_ESOP_Shares

# Now substitute New_ESOP_Shares:
# S_post * Pre_Money_Valuation / Post_Money_Valuation - (Post_Round_ESOP_Percentage * S_post) = Total_Pre_Money_Shares - Existing_ESOP_Shares
# S_post * (Pre_Money_Valuation / Post_Money_Valuation - Post_Round_ESOP_Percentage) = Total_Pre_Money_Shares - Existing_ESOP_Shares

# S_post = (Total_Pre_Money_Shares - Existing_ESOP_Shares) / (Pre_Money_Valuation / Post_Money_Valuation - Post_Round_ESOP_Percentage)
# Existing_ESOP_Shares = pre_money_shares_distribution['ESOP']

# Calculate total post-money shares based on the formula derived
# Ensure the denominator is not zero or negative
denominator = (pre_money_valuation / post_money_valuation) - post_round_esop_percentage
if denominator <= 0:
    print("Error: Cannot calculate post-money shares with the given inputs. Check valuation and ESOP percentage.")
    # Set values to None or handle error appropriately
    total_post_money_shares = None
    post_money_share_price = None
    new_round_shares = None
    new_esop_shares = None
else:
    total_post_money_shares = (total_pre_money_shares - pre_money_shares_distribution['ESOP']) / denominator
    print(f"Total Post-Money Shares (calculated): {total_post_money_shares:,.0f}")

    # 4. Calculate the post-money share price
    post_money_share_price = post_money_valuation / total_post_money_shares
    print(f"Post-Money Share Price: ${post_money_share_price:,.4f}")

    # 3. Calculate the total number of new shares issued in the round
    new_round_shares = round_size / post_money_share_price
    print(f"New Shares Issued in Round: {new_round_shares:,.0f}")

    # 7. Calculate the number of new shares for the ESOP pool
    target_post_money_esop_shares = post_round_esop_percentage * total_post_money_shares
    new_esop_shares = target_post_money_esop_shares - pre_money_shares_distribution['ESOP']
    # Ensure new_esop_shares is not negative (shouldn't be if calculation is correct, but good check)
    new_esop_shares = max(0, new_esop_shares)
    print(f"Target Post-Money ESOP Shares: {target_post_money_esop_shares:,.0f}")
    print(f"New ESOP Shares Added: {new_esop_shares:,.0f}")

    # 5. Verify total number of shares after the round (existing + new round + new ESOP)
    verified_total_post_money_shares = total_pre_money_shares + new_round_shares + new_esop_shares
    print(f"Verified Total Post-Money Shares: {verified_total_post_money_shares:,.0f}")
    # This should be very close to total_post_money_shares calculated earlier

    # 8. Update the shares for each existing shareholder group and the ESOP after the new shares
    post_money_shares_distribution = pre_money_shares_distribution.copy()
    post_money_shares_distribution['ESOP'] += new_esop_shares
    post_money_shares_distribution['New Investors'] = new_round_shares

    print("\nPost-Money Share Distribution:")
    for group, shares in post_money_shares_distribution.items():
        print(f"{group}: {shares:,.0f} shares")


    # 9. Calculate the ownership percentage for each group in the post-money cap table
    post_money_cap_table = {}
    for group, shares in post_money_shares_distribution.items():
        post_money_cap_table[group] = shares / total_post_money_shares
        print(f"Post-money ownership for {group}: {post_money_cap_table[group]:.2%}")

    # 10. Create a dictionary or similar structure to represent the post-money cap table
    # We already have post_money_cap_table dictionary

    # 11. Calculate the dilution for each existing shareholder group
    dilution_breakdown = {}
    for group, pre_money_percentage in current_cap_table.items():
        if group in post_money_cap_table:
            post_money_percentage = post_money_cap_table[group]
            dilution = pre_money_percentage - post_money_percentage
            dilution_breakdown[group] = dilution
            print(f"Dilution for {group}: {dilution:.2%}")
        else:
            # This case should not happen if the group exists in current_cap_table
            dilution_breakdown[group] = pre_money_percentage # They were completely diluted (unlikely for core groups)

    print("\nDilution Breakdown:")
    for group, dilution in dilution_breakdown.items():
         print(f"{group}: {dilution:.2%}")

# Generate Dilution Breakdown Chart
# 2. Create a list of shareholder groups from the dilution_breakdown dictionary keys.
shareholder_groups = list(dilution_breakdown.keys())

# 3. Create a list of dilution percentages from the dilution_breakdown dictionary values.
dilution_percentages = list(dilution_breakdown.values())

# 4. Create a Plotly bar chart using go.Figure() and go.Bar(). Use the list of shareholder groups for the x-axis and the list of dilution percentages for the y-axis.
fig = go.Figure(data=[go.Bar(x=shareholder_groups, y=dilution_percentages)])

# 5. Update the layout of the figure to add a title (e.g., "Dilution Breakdown by Shareholder Group"), x-axis label ("Shareholder Group"), and y-axis label ("Dilution (%)"). Format the y-axis as percentages.
fig.update_layout(
    title="Dilution Breakdown by Shareholder Group",
    xaxis_title="Shareholder Group",
    yaxis_title="Dilution (%)",
    yaxis_tickformat=".1%" # Format y-axis as percentage with one decimal place
)

# 6. Display the plot using the .show() method.
fig.show()


## Term Sheet Comparison Module

# Term Sheet Comparison Inputs
offer_scenarios = {
    'Offer A': {
        'valuation': 12_000_000,
        'round_size': 2_000_000
    },
    'Offer B': {
        'valuation': 15_000_000,
        'round_size': 3_000_000
    }
    # Add more scenarios as needed
}

print("\nDefined Offer Scenarios:")
for offer, details in offer_scenarios.items():
    print(f"- {offer}: Valuation=${details['valuation']:,.0f}, Round Size=${details['round_size']:,.0f}")

# Calculate Metrics for Each Scenario
# List to store the calculated metrics for each scenario
scenario_metrics = []

# Iterate through each offer scenario
for offer_name, details in offer_scenarios.items():
    valuation = details['valuation']
    round_size = details['round_size']

    # Calculate post-money valuation
    post_money_valuation = valuation + round_size

    # Calculate investor percentage (New Investors)
    investor_percentage = round_size / post_money_valuation

    # Assuming 'Founders' percentage is part of the existing ownership that gets diluted
    # For simplicity in this comparison module, let's calculate the founder percentage
    # relative to the pre-money ownership structure if the round happens.
    # However, the prompt asks for "founder %" in the output for the *Term Sheet Comparison*
    # module, which usually refers to their ownership *after* the round under that specific term sheet.
    # This requires knowing the *current* founder percentage before the round.
    # Let's use the 'Founders' percentage from the previously defined `current_cap_table`.
    # The founder percentage *after* the round would be their pre-money percentage * (Pre-Money Valuation / Post-Money Valuation)
    # assuming no new shares are issued to founders and the ESOP is handled separately.
    # However, the prompt asks for 'founder %' in the context of the offer itself,
    # which is a bit ambiguous. Let's interpret 'founder %' here as their percentage
    # of the company *after* the round, based on their initial percentage being diluted
    # by the new money and potentially a new ESOP pool (though the ESOP calculation
    # was done in the previous module and isn't explicitly tied to each offer here).
    # A simpler interpretation for a *Term Sheet Comparison* might be the percentage
    # the *new investors* get vs. the percentage *retained by existing shareholders*,
    # where founders are a part of the existing shareholders.
    # Let's calculate the percentage retained by *all* existing shareholders,
    # which is (Pre-Money Valuation / Post-Money Valuation). The founder percentage
    # *within* the existing group remains the same relative to other existing shareholders,
    # but their overall percentage of the company decreases.
    # Let's calculate the percentage retained by existing shareholders.
    existing_shareholder_percentage_retained = valuation / post_money_valuation

    # We can also show the founders' specific percentage if we use the current_cap_table['Founders']
    # founder_percentage_post_round = current_cap_table['Founders'] * existing_shareholder_percentage_retained

    # Let's focus on the split between New Investors and Existing Shareholders as this is common in term sheet summaries.
    # The prompt specifically asks for "founder %" and "investor %".
    # Let's assume "investor %" means the new investors, and "founder %" means the founders' percentage *post-money*.
    # This requires using the current_cap_table['Founders'] from the previous module.
    founder_percentage_pre_money = current_cap_table['Founders']
    founder_percentage_post_money = founder_percentage_pre_money * (valuation / post_money_valuation)


    # Store the metrics for the current scenario
    scenario_metrics.append({
        'Offer': offer_name,
        'Pre-Money Valuation': valuation,
        'Round Size': round_size,
        'Post-Money Valuation': post_money_valuation,
        'New Investor Percentage': investor_percentage,
        'Founder Percentage (Post-Money)': founder_percentage_post_money # Using calculated post-money founder %
    })

# Print the calculated metrics for each offer scenario
print("\nTerm Sheet Comparison Metrics:")
for metrics in scenario_metrics:
    print(f"\n{metrics['Offer']}:")
    print(f"  Pre-Money Valuation: ${metrics['Pre-Money Valuation']:,.0f}")
    print(f"  Round Size: ${metrics['Round Size']:,.0f}")
    print(f"  Post-Money Valuation: ${metrics['Post-Money Valuation']:,.0f}")
    print(f"  New Investor Percentage: {metrics['New Investor Percentage']:.2%}")
    print(f"  Founder Percentage (Post-Money): {metrics['Founder Percentage (Post-Money)']:.2%}")


# Generate Term Sheet Comparison Chart
# Create a list of offer names
offer_names = [scenario['Offer'] for scenario in scenario_metrics]

# Create lists for metrics
post_money_valuations = [scenario['Post-Money Valuation'] for scenario in scenario_metrics]
new_investor_percentages = [scenario['New Investor Percentage'] for scenario in scenario_metrics]
founder_percentages = [scenario['Founder Percentage (Post-Money)'] for scenario in scenario_metrics]

# Create a Plotly figure
fig = go.Figure()

# Add bar chart traces for each metric
fig.add_trace(go.Bar(
    x=offer_names,
    y=post_money_valuations,
    name='Post-Money Valuation',
    yaxis='y1' # Assign to the primary y-axis
))

fig.add_trace(go.Bar(
    x=offer_names,
    y=new_investor_percentages,
    name='New Investor Percentage',
    yaxis='y2' # Assign to the secondary y-axis
))

fig.add_trace(go.Bar(
    x=offer_names,
    y=founder_percentages,
    name='Founder Percentage (Post-Money)',
    yaxis='y2' # Assign to the secondary y-axis
))

# Update the layout for a grouped bar chart with two y-axes
fig.update_layout(
    title="Term Sheet Comparison",
    xaxis_title="Offer Scenario",
    yaxis=dict(
        title="Post-Money Valuation ($)",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="Percentage",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right",
        tickformat=".1%" # Format as percentage
    ),
    barmode='group', # Group bars for each offer
    legend=dict(x=0, y=1.1, orientation="h") # Position legend
)

# Display the plot
fig.show()

"""## Refine and document

### Subtask:
Add comments and documentation to the code.

**Reasoning**:
Add markdown cells for module descriptions and chart explanations, and add comments to the code for clarity and input labeling.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Revenue Forecast Module ---

# ## Revenue Forecast
# This module forecasts the company's revenue over a 24-month period based on key growth drivers.
#
# **Inputs:**
# - Starting Revenue: Initial monthly revenue.
# - Monthly Growth Rate: Expected month-over-month revenue growth percentage.
# - Churn Rate: Percentage of customers lost each month.
# - ARPU (Average Revenue Per User): Average revenue generated per customer per month.
# - Scenario Toggles: Ability to select different sets of input assumptions (Base Case, Optimistic, Pessimistic).
#
# **Outputs:**
# - 24-Month Revenue Forecast: A list of projected monthly revenue values.
# - Revenue Forecast Chart: A visual representation of the forecast over time.

# Revenue Forecast Inputs - Editable Section
# Define different scenario inputs. Modify these values to change growth assumptions.
scenarios = {
    "Base Case": {
        "starting_revenue": 10000,  # Initial monthly revenue in USD
        "monthly_growth_rate": 0.05,  # 5% monthly revenue growth rate (as a decimal)
        "churn_rate": 0.02,  # 2% monthly churn rate (as a decimal)
        "arpu": 50  # Average Revenue Per User in USD
    },
    "Optimistic": {
        "starting_revenue": 12000,
        "monthly_growth_rate": 0.07,
        "churn_rate": 0.01,
        "arpu": 55
    },
    "Pessimistic": {
        "starting_revenue": 8000,
        "monthly_growth_rate": 0.03,
        "churn_rate": 0.03,
        "arpu": 45
    }
}

# Select the scenario to use for the forecast. Change the string value below.
scenario = "Base Case"  # Options: "Base Case", "Optimistic", "Pessimistic"

# Get parameters for the selected scenario
selected_scenario_params = scenarios[scenario]
starting_revenue = selected_scenario_params["starting_revenue"]
monthly_growth_rate = selected_scenario_params["monthly_growth_rate"]
churn_rate = selected_scenario_params["churn_rate"]
arpu = selected_scenario_params["arpu"]

# Revenue Forecast Calculation
# Create a list to store the monthly revenue forecast
revenue_forecast = []

# Initialize the first month's revenue
revenue_forecast.append(starting_revenue)

# Calculate starting users from starting revenue and ARPU
starting_users = starting_revenue / arpu
projected_users = starting_users

# Iterate through the subsequent 23 months (total 24 months)
for month in range(1, 24):
    # Calculate the projected users for the current month
    # Users grow by (1 + growth_rate) and decrease by (1 - churn_rate)
    projected_users = projected_users * (1 + monthly_growth_rate) * (1 - churn_rate)
    # Calculate the revenue for the current month (projected users * ARPU)
    current_month_revenue = projected_users * arpu
    # Append the calculated monthly revenue to the forecast list
    revenue_forecast.append(current_month_revenue)

print(f"24-Month Revenue Forecast ({scenario} Scenario) Calculated.")
# print(revenue_forecast) # Uncomment to print the full list

# Generate Revenue Forecast Chart
# ## Revenue Forecast Chart
# This chart visualizes the projected monthly revenue over the next 24 months
# based on the selected scenario inputs.
#
# **Interpretation:** Observe the trend of revenue growth over time under
# the chosen assumptions. Compare different scenarios by changing the 'scenario' input.

# Create a list of month numbers from 1 to 24 to represent the x-axis.
months = list(range(1, 25))

# Create a Plotly figure.
fig = go.Figure()

# Add a scatter plot trace for the revenue forecast.
fig.add_trace(go.Scatter(x=months, y=revenue_forecast, mode='lines+markers', name=scenario))

# Update the layout of the figure.
fig.update_layout(
    title="24-Month Revenue Forecast",
    xaxis_title="Month",
    yaxis_title="Revenue ($)"
)

# Display the plot.
fig.show()


# --- Unit Economics Stress Test Module ---

# ## Unit Economics Stress Test
# This module analyzes key unit economics metrics (LTV/CAC Ratio and Contribution Margin)
# and provides a stress grid to show how changes in CAC and LTV impact these metrics.
#
# **Inputs:**
# - CAC (Customer Acquisition Cost): Cost to acquire one customer.
# - LTV (Customer Lifetime Value): Total revenue expected from a single customer.
# - Fixed Cost Per Unit: Fixed cost associated with delivering the product/service to one unit/customer.
# - Variable Cost Per Unit: Variable cost associated with delivering the product/service to one unit/customer.
#
# **Outputs:**
# - LTV/CAC Ratio: Calculated ratio of LTV to CAC.
# - Contribution Margin Per Unit: Calculated profit per unit after variable and fixed costs directly associated with the unit.
# - Stress Grid: A table showing LTV/CAC Ratio and Contribution Margin for a range of CAC and LTV values.

# Unit Economics Stress Test Inputs - Editable Section
cac = 100  # Customer Acquisition Cost in USD
ltv = 500  # Customer Lifetime Value in USD
fixed_cost_per_unit = 10  # Fixed cost associated with each unit in USD
variable_cost_per_unit = 20  # Variable cost associated with each unit in USD

# Note: ARPU is used from the Revenue Forecast module inputs for Contribution Margin calculation.

# Calculate Unit Economics Metrics
# Calculate LTV to CAC ratio
ltv_cac_ratio = ltv / cac

# Calculate contribution margin per unit
# Contribution Margin = ARPU - (Fixed Cost Per Unit + Variable Cost Per Unit)
contribution_margin_per_unit = arpu - (fixed_cost_per_unit + variable_cost_per_unit)

# Print the calculated values
print(f"\n--- Unit Economics Metrics ---")
print(f"LTV/CAC Ratio: {ltv_cac_ratio:.2f}")
print(f"Contribution Margin Per Unit: ${contribution_margin_per_unit:.2f}")

# Create Stress Grid
# ## Unit Economics Stress Grid
# This table shows the calculated LTV/CAC Ratio and Contribution Margin Per Unit
# across a grid of varying Customer Acquisition Cost (CAC) and Customer Lifetime Value (LTV) inputs.
#
# **Interpretation:** Use this grid to understand the sensitivity of your unit economics
# to changes in the costs of acquiring customers and the revenue generated from them.
# Identify the combinations of CAC and LTV that result in desired LTV/CAC ratios (e.g., > 3x).

# Define ranges for CAC and LTV for the stress grid. Modify these ranges and the number of steps.
cac_range = np.linspace(50, 200, 10) # CAC from 50 to 200 in 10 equal steps
ltv_range = np.linspace(200, 800, 10) # LTV from 200 to 800 in 10 equal steps

# Create a list to store the results
stress_test_results = []

# Iterate through all combinations of CAC and LTV in the defined ranges
for current_cac in cac_range:
    for current_ltv in ltv_range:
        # Calculate LTV/CAC ratio for the current combination
        current_ltv_cac_ratio = current_ltv / current_cac

        # Store the results in a dictionary
        result = {
            'CAC': current_cac,
            'LTV': current_ltv,
            'LTV/CAC Ratio': current_ltv_cac_ratio,
            # Contribution Margin Per Unit is constant across this grid as it doesn't depend on CAC or LTV
            'Contribution Margin Per Unit': contribution_margin_per_unit
        }

        # Append the dictionary to the results list
        stress_test_results.append(result)

# Convert the list of results into a pandas DataFrame for easy viewing
stress_grid_df = pd.DataFrame(stress_test_results)

# Display the stress grid DataFrame
print("\n--- Unit Economics Stress Grid (LTV/CAC Ratio and Contribution Margin) ---")
display(stress_grid_df)


# --- Cap Table Simulator Module ---

# ## Cap Table Simulator
# This module simulates the impact of a funding round on the company's capitalization table,
# calculating post-money ownership percentages, share price, and dilution for existing shareholders.
#
# **Inputs:**
# - Pre-Money Valuation: The company's valuation before the new funding round.
# - Round Size: The total amount of capital being raised in the round.
# - Current Cap Table: A breakdown of existing equity ownership percentages (Founders, Pre-Seed Investors, ESOP, etc.).
# - Post-Round ESOP %: The target percentage of the company's equity allocated to the ESOP pool *after* the funding round.
#
# **Outputs:**
# - Post-Money Valuation: The company's valuation after the funding round.
# - Post-Money Share Price: The price per share at which the new investment is made.
# - Post-Money Cap Table: A breakdown of ownership percentages for all shareholder groups after the round (including new investors).
# - Dilution Breakdown: The percentage decrease in ownership for each existing shareholder group.
# - Dilution Breakdown Chart: A visual comparison of dilution across existing shareholder groups.

# Cap Table Simulator Inputs - Editable Section
pre_money_valuation = 10_000_000  # Pre-money valuation of the company in USD
round_size = 2_000_000  # Total funding amount being raised in USD

# Current Cap Table (represented as a dictionary of equity ownership percentages). Modify percentages and groups as needed.
current_cap_table = {
    'Founders': 0.70,       # 70% ownership before the round
    'Pre-Seed Investors': 0.20, # 20% ownership before the round
    'ESOP': 0.10            # 10% ownership in the existing ESOP pool before the round
}

# Desired post-round ESOP percentage. This is the target size of the ESOP pool *after* the round.
post_round_esop_percentage = 0.15 # Target 15% of the company's equity post-money

# Calculate Post-Money Cap Table, Share Price, and Dilution

# 1. Calculate the post-money valuation
post_money_valuation = pre_money_valuation + round_size
print(f"\n--- Cap Table Simulation ---")
print(f"Post-Money Valuation: ${post_money_valuation:,.2f}")

# 2. Assume a notional pre-money share price (e.g., $1) to calculate a total pre-money share count.
# This is a common approach when starting with valuations and percentages rather than a fixed share count.
notional_pre_money_share_price = 1.0
total_pre_money_shares = pre_money_valuation / notional_pre_money_share_price
print(f"Total Notional Pre-Money Shares: {total_pre_money_shares:,.0f}")

# 3. Calculate the number of shares held by each existing shareholder group before the round
pre_money_shares_distribution = {}
for group, percentage in current_cap_table.items():
    pre_money_shares_distribution[group] = total_pre_money_shares * percentage
    print(f"Pre-money shares for {group}: {pre_money_shares_distribution[group]:,.0f}")

# Calculate Total Post-Money Shares, Share Price, New Round Shares, and New ESOP Shares
# This involves solving a system of equations to account for the target post-round ESOP percentage.
# The formula used here calculates the total post-money shares required to satisfy
# the pre-money shareholders (excluding existing ESOP) and the target post-round ESOP percentage,
# scaled by the ratio of pre-money to post-money valuation.

existing_esop_shares_pre_money = pre_money_shares_distribution.get('ESOP', 0) # Get existing ESOP shares, default to 0 if group doesn't exist

# Calculate the denominator for the total post-money shares formula
# Denominator = (Pre-Money Valuation / Post-Money Valuation) - Post-Round ESOP Percentage
denominator = (pre_money_valuation / post_money_valuation) - post_round_esop_percentage

# Check if the denominator is valid (non-positive denominator would indicate an issue with inputs)
if denominator <= 0:
    print("Error: Cannot calculate post-money shares with the given inputs.")
    print("Ensure Pre-Money Valuation / Post-Money Valuation is greater than the Post-Round ESOP Percentage.")
    # Set calculated values to None or handle error as appropriate
    total_post_money_shares = None
    post_money_share_price = None
    new_round_shares = None
    new_esop_shares = None
else:
    # Calculate total post-money shares based on the formula derived from solving the system of equations
    # Total Post-Money Shares = (Total Pre-Money Shares - Existing ESOP Shares) / ((Pre-Money Valuation / Post-Money Valuation) - Post-Round ESOP Percentage)
    total_post_money_shares = (total_pre_money_shares - existing_esop_shares_pre_money) / denominator
    print(f"Total Post-Money Shares (calculated): {total_post_money_shares:,.0f}")

    # 4. Calculate the post-money share price
    # Post-Money Share Price = Post-Money Valuation / Total Post-Money Shares
    post_money_share_price = post_money_valuation / total_post_money_shares
    print(f"Post-Money Share Price: ${post_money_share_price:,.4f}")

    # 5. Calculate the total number of new shares issued to investors in the round
    # New Round Shares = Round Size / Post-Money Share Price
    new_round_shares = round_size / post_money_share_price
    print(f"New Shares Issued to Investors: {new_round_shares:,.0f}")

    # 6. Calculate the number of new shares that need to be added to the ESOP pool
    # Target Post-Money ESOP Shares = Post-Round ESOP Percentage * Total Post-Money Shares
    target_post_money_esop_shares = post_round_esop_percentage * total_post_money_shares
    # New ESOP Shares = Target Post-Money ESOP Shares - Existing ESOP Shares
    new_esop_shares = target_post_money_esop_shares - existing_esop_shares_pre_money
    # Ensure new_esop_shares is not negative (can happen if target ESOP is smaller than existing)
    new_esop_shares = max(0, new_esop_shares) # Assume we don't reduce the ESOP pool
    print(f"Target Post-Money ESOP Shares: {target_post_money_esop_shares:,.0f}")
    print(f"New ESOP Shares Added: {new_esop_shares:,.0f}")

    # 7. Verify total number of shares after the round (existing + new round + new ESOP)
    verified_total_post_money_shares = total_pre_money_shares + new_round_shares + new_esop_shares
    # This value should be very close to the total_post_money_shares calculated earlier due to floating point precision.
    print(f"Verified Total Post-Money Shares: {verified_total_post_money_shares:,.0f}")


    # 8. Update the shares for each existing shareholder group and add the new investors and new ESOP shares
    post_money_shares_distribution = pre_money_shares_distribution.copy()
    # Add the new ESOP shares to the ESOP pool
    post_money_shares_distribution['ESOP'] = existing_esop_shares_pre_money + new_esop_shares
    # Add the new investors as a group
    post_money_shares_distribution['New Investors'] = new_round_shares

    print("\nPost-Money Share Distribution:")
    for group, shares in post_money_shares_distribution.items():
        print(f"{group}: {shares:,.0f} shares")


    # 9. Calculate the ownership percentage for each group in the post-money cap table
    post_money_cap_table = {}
    # Ensure we use the calculated total_post_money_shares for percentages
    final_total_shares = total_post_money_shares if total_post_money_shares is not None else verified_total_post_money_shares # Use the calculated one if available
    if final_total_shares is not None and final_total_shares > 0:
        for group, shares in post_money_shares_distribution.items():
            post_money_cap_table[group] = shares / final_total_shares
            print(f"Post-money ownership for {group}: {post_money_cap_table[group]:.2%}")
    else:
        print("Cannot calculate post-money percentages: Total post-money shares is zero or invalid.")
        post_money_cap_table = None


    # 10. Calculate the dilution for each existing shareholder group
    dilution_breakdown = {}
    # Only calculate dilution if post_money_cap_table was successfully created
    if post_money_cap_table is not None:
        print("\nDilution Breakdown:")
        for group, pre_money_percentage in current_cap_table.items():
            # Dilution = Pre-Money Percentage - Post-Money Percentage
            if group in post_money_cap_table:
                post_money_percentage = post_money_cap_table[group]
                dilution = pre_money_percentage - post_money_percentage
                dilution_breakdown[group] = dilution
                print(f"Dilution for {group}: {dilution:.2%}")
            else:
                 # This case means an existing group is somehow not in the post-money cap table (unlikely for core groups)
                 dilution_breakdown[group] = pre_money_percentage # Treat as 100% dilution if not found
                 print(f"Dilution for {group}: {pre_money_percentage:.2%} (Group not found in post-money cap table)")
    else:
        print("Cannot calculate dilution breakdown: Post-money cap table is not available.")
        dilution_breakdown = None


# Generate Dilution Breakdown Chart
# ## Dilution Breakdown Chart
# This bar chart visually represents the percentage of ownership diluted for each
# existing shareholder group as a result of the funding round and ESOP expansion.
#
# **Interpretation:** Understand which shareholder groups experienced the most
# dilution and by how much. Note that a negative dilution percentage for the
# ESOP indicates an increase in its percentage ownership post-round to reach the target size.

# Ensure dilution_breakdown was successfully calculated before attempting to plot
if dilution_breakdown is not None:
    # Create a list of shareholder groups from the dilution_breakdown dictionary keys.
    shareholder_groups = list(dilution_breakdown.keys())

    # Create a list of dilution percentages from the dilution_breakdown dictionary values.
    dilution_percentages = list(dilution_breakdown.values())

    # Create a Plotly bar chart.
    fig = go.Figure(data=[go.Bar(x=shareholder_groups, y=dilution_percentages)])

    # Update the layout of the figure.
    fig.update_layout(
        title="Dilution Breakdown by Shareholder Group",
        xaxis_title="Shareholder Group",
        yaxis_title="Dilution (%)",
        yaxis_tickformat=".1%" # Format y-axis as percentage with one decimal place
    )

    # Display the plot.
    fig.show()
else:
    print("Dilution breakdown chart cannot be generated due to calculation errors.")


# --- Term Sheet Comparison Module ---

# ## Term Sheet Comparison
# This module allows you to compare different funding offer scenarios (term sheets)
# side-by-side based on key financial outcomes like post-money valuation,
# new investor ownership percentage, and founder ownership percentage.
#
# **Inputs:**
# - Offer Scenarios: A dictionary defining multiple term sheet offers, each with a specified
#   pre-money valuation and round size.
#
# **Outputs:**
# - Calculated Metrics: A summary of Post-Money Valuation, New Investor Percentage,
#   and Founder Percentage (Post-Money) for each offer scenario.
# - Comparison Chart: A side-by-side bar chart visualizing these metrics across scenarios.

# Term Sheet Comparison Inputs - Editable Section
# Define 2-3 (or more) offer scenarios. Modify the valuations and round sizes for each offer.
offer_scenarios = {
    'Offer A': {
        'valuation': 12_000_000, # Pre-money valuation for Offer A in USD
        'round_size': 2_000_000 # Round size for Offer A in USD
    },
    'Offer B': {
        'valuation': 15_000_000, # Pre-money valuation for Offer B in USD
        'round_size': 3_000_000 # Round size for Offer B in USD
    }
    # Add more scenarios by adding new key-value pairs, e.g.:
    # 'Offer C': {
    #     'valuation': 13_500_000,
    #     'round_size': 2_500_000
    # }
}

print("\n--- Defined Offer Scenarios ---")
for offer, details in offer_scenarios.items():
    print(f"- {offer}: Valuation=${details['valuation']:,.0f}, Round Size=${details['round_size']:,.0f}")

# Calculate Metrics for Each Scenario
# List to store the calculated metrics for each scenario
scenario_metrics = []

# Iterate through each offer scenario defined in the inputs
for offer_name, details in offer_scenarios.items():
    valuation = details['valuation']
    round_size = details['round_size']

    # Calculate post-money valuation for the current offer
    post_money_valuation = valuation + round_size

    # Calculate investor percentage (percentage owned by new investors)
    investor_percentage = round_size / post_money_valuation

    # Calculate the founder percentage *after* this specific round scenario.
    # We use the founder's pre-money percentage from the Cap Table module inputs.
    # Founder Post-Money % = Founder Pre-Money % * (Pre-Money Valuation / Post-Money Valuation)
    # This assumes founder shares are not changing, only being diluted by new shares.
    # Note: This calculation simplifies the Cap Table logic and doesn't account for ESOP changes specific to this offer.
    # It shows dilution impact on founders from *this* offer's terms compared to their *initial* percentage.
    founder_percentage_pre_money = current_cap_table.get('Founders', 0) # Get founder % from Cap Table inputs, default to 0
    founder_percentage_post_money = founder_percentage_pre_money * (valuation / post_money_valuation)


    # Store the calculated metrics for the current scenario in a dictionary
    scenario_metrics.append({
        'Offer': offer_name,
        'Pre-Money Valuation': valuation,
        'Round Size': round_size,
        'Post-Money Valuation': post_money_valuation,
        'New Investor Percentage': investor_percentage,
        'Founder Percentage (Post-Money)': founder_percentage_post_money # Founder % after this specific offer
    })

# Print the calculated metrics for each offer scenario
print("\n--- Term Sheet Comparison Metrics ---")
for metrics in scenario_metrics:
    print(f"\n{metrics['Offer']}:")
    print(f"  Pre-Money Valuation: ${metrics['Pre-Money Valuation']:,.0f}")
    print(f"  Round Size: ${metrics['Round Size']:,.0f}")
    print(f"  Post-Money Valuation: ${metrics['Post-Money Valuation']:,.0f}")
    print(f"  New Investor Percentage: {metrics['New Investor Percentage']:.2%}")
    print(f"  Founder Percentage (Post-Money): {metrics['Founder Percentage (Post-Money)']:.2%}")


# Generate Term Sheet Comparison Chart
# ## Term Sheet Comparison Chart
# This bar chart compares key metrics (Post-Money Valuation, New Investor Percentage,
# and Founder Percentage Post-Money) across different offer scenarios side-by-side.
#
# **Interpretation:** Use this chart to visually compare the trade-offs between
# different term sheet offers. Higher post-money valuations might result in
# lower investor percentages but also less dilution for founders.

# Create lists of data for the chart from the calculated scenario_metrics
offer_names = [scenario['Offer'] for scenario in scenario_metrics]
post_money_valuations = [scenario['Post-Money Valuation'] for scenario in scenario_metrics]
new_investor_percentages = [scenario['New Investor Percentage'] for scenario in scenario_metrics]
founder_percentages = [scenario['Founder Percentage (Post-Money)'] for scenario in scenario_metrics]

# Create a Plotly figure
fig = go.Figure()

# Add bar chart traces for each metric
# Post-Money Valuation bar (using primary y-axis)
fig.add_trace(go.Bar(
    x=offer_names,
    y=post_money_valuations,
    name='Post-Money Valuation',
    yaxis='y1' # Assign to the primary y-axis (left)
))

# New Investor Percentage bar (using secondary y-axis)
fig.add_trace(go.Bar(
    x=offer_names,
    y=new_investor_percentages,
    name='New Investor %',
    yaxis='y2' # Assign to the secondary y-axis (right)
))

# Founder Percentage (Post-Money) bar (also using secondary y-axis)
fig.add_trace(go.Bar(
    x=offer_names,
    y=founder_percentages,
    name='Founder % (Post-Money)',
    yaxis='y2' # Assign to the secondary y-axis (right)
))

# Update the layout for a grouped bar chart with two y-axes
fig.update_layout(
    title="Term Sheet Comparison",
    xaxis_title="Offer Scenario",
    # Configuration for the primary y-axis (Valuation)
    yaxis=dict(
        title="Post-Money Valuation ($)",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
        side="left", # Position primary y-axis on the left
        tickformat="$,.0f" # Format as currency with commas and no decimals
    ),
    # Configuration for the secondary y-axis (Percentages)
    yaxis2=dict(
        title="Percentage",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y", # Overlay on the primary y-axis
        side="right", # Position secondary y-axis on the right
        tickformat=".1%" # Format as percentage with one decimal place
    ),
    barmode='group', # Group bars for each offer scenario
    legend=dict(x=0, y=1.1, orientation="h") # Position legend horizontally above the plot
)

# Display the plot
fig.show()

"""## Summary:

## Data Analysis Key Findings

*   The financial model was successfully built in Python using `pandas` and `plotly`, incorporating modules for Revenue Forecast, Unit Economics Stress Test, Cap Table Simulator, and Term Sheet Comparison.
*   The Revenue Forecast module includes editable inputs for starting revenue, monthly growth, churn, and ARPU, calculates a 24-month forecast, supports scenario toggles ("Base Case", "Optimistic", "Pessimistic"), and visualizes the forecast with a Plotly chart.
*   The Unit Economics Stress Test module includes inputs for CAC, LTV, and fixed/variable costs per unit, calculates the LTV/CAC ratio and contribution margin per unit, and generates a `pandas` DataFrame stress grid showing LTV/CAC and contribution margin across ranges of CAC and LTV.
*   The Cap Table Simulator module takes inputs for pre-money valuation, round size, current cap table percentages (Founders, Pre-Seed, ESOP), and desired post-round ESOP percentage, calculates the post-money valuation, share price, post-money share distribution, post-money ownership percentages, and dilution breakdown, and visualizes dilution with a Plotly bar chart.
*   The Term Sheet Comparison module allows defining multiple offer scenarios with valuation and round size inputs, calculates the post-money valuation, new investor percentage, and founder post-money percentage for each, and provides a side-by-side Plotly bar chart for visual comparison.
*   All modules were successfully integrated into a single Colab-ready notebook with clear sections, editable inputs, and visual outputs.

## Insights or Next Steps

*   The current Cap Table Simulator assumes a notional pre-money share price ($1) to derive share counts. A next step could be to allow the user to input an actual current total share count or a known pre-money share price for greater accuracy if that data is available.
*   While the Term Sheet Comparison calculates founder percentage post-money based on pre-money percentage and dilution, it doesn't fully integrate the ESOP pool expansion logic specific to each offer scenario. A potential enhancement is to calculate the full post-money cap table *for each offer* to get a more precise breakdown, including the ESOP impact per offer.

"""
