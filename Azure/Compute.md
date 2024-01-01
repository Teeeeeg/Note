# Cloud Models
![](img/cloud-models.png)
# Availability Concepts
## Fault Domain
Basically servers in a rack.
## Update Domain
Logical group of hardware that can undergo maintenance at the same time.
## Availability Set
A collection of Fault domain and Update domain the VMs will be spread across.
## Availability Zone
A physically separate zone within an Azure region.
# App Services
A fully managed web hosting for websites.  
Integrates with many source controls.
## Auto Scaling
Configure the app services to scale in or out.
# Containers
Thin packaging model.  
Packages software, its dependency and configuration files.
## Docker
Container environment
## Kubernetes
Container management
# Azure Functions
Small, focused functions running as a result of an event.  
Great for event driven systems.  
Managed by Azure.  
## Triggers
The event that made the function run.
## Bindings
Declarative connection to other resources.  
Input or output.  
## Cold start
Azure functions are completely managed by Azure.   
Azure will take down the Function's host.  
The next activation will take time.