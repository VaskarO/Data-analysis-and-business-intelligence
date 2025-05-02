# Overview
This project models, simulates, and normalizes a data warehouse pipeline for a farm producing and selling multiple crops. The pipeline is designed to support robust data analytics and visualization, focusing on both production efficiency and sales performance.

## 1. Initial Table Design (Denormalized)
We began by defining denormalized (flat) fact tables for quick prototyping and straightforward data access. Two core tables were defined:

### fact_production (Denormalized)
Captures production-level details for each crop and field, including:

Crop variety and yield

Fertilizer and pesticide usage

Irrigation method

Labor and machinery details

### fact_sales (Denormalized)
Captures sales transactions with:

Buyer and region info

Channel and pricing

Discounts, shipping, profit margin

Actual vs market price comparison

These tables were intentionally flat to simplify prototyping and visual dashboard integration.

## Dummy Data Generation
To simulate realistic conditions, we used Python with the Faker, pandas, and numpy libraries to generate artificial data:

15000 production records

7500 sales records (1-many with production)


# Data Normalization & Star Schema
To enable better long-term analytics and maintainability, we normalized both fact tables into a professional star schema, supporting:

Query optimization

Data integrity

Scalability

Easier reporting in tools like Power BI, Tableau

## Dimension Tables
dim_crop: Type & variety of crops

dim_field: Farm & field details

dim_input: Fertilizer, pesticide, irrigation specs

dim_environment: Soil pH, rainfall, temperature

dim_labor: Workers, hours, machinery

dim_date: Shared calendar table for joins

dim_buyer: Buyer profiles and regions

dim_sales_channel: Sales method (Direct, Online, etc.)

## Fact Tables
fact_production: Linked to field, crop, input, env, labor, date

fact_sales: Linked to production, crop, buyer, channel, date

Foreign keys enforce integrity, and surrogate IDs allow deduplication and reusability of dimension attributes.