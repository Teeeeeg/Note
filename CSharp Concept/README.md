# Covariance and Contravariance
-   `Covariance`  
    Enables you to use a more **derived** type than originally specified.
-   `Contravariance`  
    Enables you to use a more **generic** (less derived) type than originally specified.
    
```CSharp
// Covariance. 
IEnumerable<string> strings = new List<string>(); 
// An object that is instantiated with a more derived type argument 
// is assigned to an object instantiated with a less derived type argument. 
// Assignment compatibility is preserved. 
IEnumerable<object> objects = strings; 

// Contravariance. 
// Assume that the following method is in the class: 
static void SetObject(object o) { } 
Action<object> actObject = SetObject; 
// An object that is instantiated with a less derived type argument 
// is assigned to an object instantiated with a more derived type argument. 
// Assignment compatibility is reversed. 
Action<string> actString = actObject;
```
`out T` specifies that the `T` is covariant. `T` will be the output type.   
`in T` specifies that the `T` is contravariant. `T` will be the input type.   
They could only be used in **generic interfaces** and **delegates**.   