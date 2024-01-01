# Dependency Injection
## Validate Scopes
`IServiceCollection` is the DI container.  
When build `IServiceProvider` from the container, it has an option `ServiceProviderOptions` with two properties:  
`ValidateOnBuild`: Throw exception if failed on build.  
`ValidateScopes`: Prevent a singleton service refers scoped services which occurs unexpected scoped services.
## Add Generics to container
Example:  
`AddTransient(typeof(IFoobar<,>), typeof(Foobar<,>))`
## Constructor Resolver
The final constructor chosen is the superset of other constructors.  
Example 1:

```
    public class Qux : IQux
    {
        public Qux(IFoo foo)
        public Qux(IFoo foo, IBar bar)           
    }
```
> The Second constructor will be chosen.

Example 2:

```
    public class Qux : IQux
    {
        public Qux(IFoo foo, IBar bar) { }
        public Qux(IBar bar, IBaz baz) { }
    }
```
> There is no superset relationship of the two constructors, so it will be an exception.  

# Configuration System
## Adding Source
For multiple configurations, the latter will cover the former.   
AddJsonFile(path, optional, reloadOnChange)

 ```
.AddJsonFile("appsettings.json", false)
.AddJsonFile($"appsettings.{environment}.json", true)
```

`appsettings.{environment}.json` will substitute `appsettings.json` but it is optional.  

`reloadOnChange` specify whether reload on change.  
Use `GetReloadToken()` for further development.

```
ChangeToken.OnChange(() => config.GetReloadToken(), () =>
{
}
```

## Structure of Configuration system 
Structure of a tree  
`IConfigurationRoot`: root  
`IConfigurationSection`: section  
`IConfiguration`: node  

## Option Pattern
Option pattern uses classes to provide strongly typed access to settings **while consuming options with dependency injection**.

`IOptions<>`: is registered as singleton.  
> Can be injected into any service lifetime.

`IOptionsSnapshot<>`: is registered as scoped which will change with every request.  
> Is useful in scenarios where options should be recomputed on every request.

`IOptionsMonitor<>`: Is registered as a Singleton and can be injected into any service lifetime.  
> Is used to retrieve options and manage options notifications for instances.

`IOptionsMonitorCache<TOptions>`: Cache instance for performance.  

`IOptionsManager<>`: is the instance on service. 
 
`Configure<TDep>(Action<TOptions>)`: Registers an action used to configure a particular type of options.  

`PostConfigure<TDep>(Action<TOptions>)`: Register an action used to post configure a particular type of option. These are run after `Configure<TDep>(Action<TOptions>)`

# Pipeline
## Write middleware
> Middleware is `Func<RequestDelegate, RequestDelegate>`  
The input is the `RequestDelegate` composed of next middlewares. Becasue after processing current middleware, it mostly needs to pass the result to the next.  
The output is a `RequestDelegate` composed with itself too. It merges to the next middlewares as a result. 

1. Implement the `IMiddleware` interface. The middleware need to register in the services. In this way, the middleware can register in any lifetime.
2. Follow the rule:
	1. A constructor with a `RequestDelegate` stands for next.
	2. A async task named `InvokeAsync`.
	
    In this way it doesn't need to register in services.    
    Instead, Use `UseMiddleware()` to register. It will be registered in singleton only.



