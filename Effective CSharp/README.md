# C # Language Idioms
## Item 1: Prefer Implicitly typed local variables
Variable name should be the information.

## Item 2: Prefer readonly to const
`readonly` is a runtime constant. It must be initiated in declaration or constructor.  
`const` is a compile-time constant. The `const` variable is replaced as the actual value. It is used when a value does not change from release to release.

```
public class UsefulValues
   {
       public static readonly int StartValue = 5;
       public const int EndValue = 10;
   }
```
>The output is 5-10. 

Distribute the Infrastructure assembly without rebuilding your Application assembly.  

```
public class UsefulValues
   {
       public static readonly int StartValue = 105;
       public const int EndValue = 120;
   }
```
> The output supposes to be 115-120. `StartValue` is resolved in runtime while `EndValue` is still 10. Because `const` variable will be replaced as the actual value. 

## Item 3: Prefer the `is` or `as` operator to casts

## Item 4: Replace string.Format() with Interpolated Strings

## Item 5: Prefer `FormattableString` for Culture-Specific Strings
```
var third =
   $"It’s the {DateTime.Now.Day} of the {DateTime.Now.Month} month";
```
> The complier will generate different code based on the compile-time type of the output being requested.  
> The code that generates a string will format that string based on the current culture on the machine where the code is executed.

## Item 6: Avoid String-ly Typed APIs
Replacing the hard-coded text with `nameof` ensures that the names match even after any rename operation.  
## Item 7: Express callbacks with delegates
Delegates can ease requirements in client classes. Callbacks can be implemented even with multicasts.
## Item 8: Use the null conditional operator for event invocations
```
 public void RaiseUpdates()
   {
       counter++;
       Updated?.Invoke(this, counter);
  
}
```
> Separating handler with some logics by using `?.Invoke()` is thread safe.

## Item 9: Minimize boxing and unboxing
The boxing and unboxing operations make copies where you might not expect. That causes bugs. 
## Item 10: Use the `new` modifier only to react to base class updates
```
 public class MyWidget : BaseWidget
   {
       public new void NormalizeValues()
       {
           // details elided.
} }
```

> In the case, `MyWidget` won't react if the `BaseWidget` updates.

Basically when the base class cause collisions in your class the `new` may come to handy.  
So use the `new` with caution.

# .NET Resource Management

## Item 11: Understand .NET resource management
Understand the GC for memory management.

## Item 12: Prefer member initializers to assignment statement
Classes have more than one constructors. It's easy for the member variables and the constructors to get out of sync.  
Using this syntax means that you cannot forget to add the proper initialization when you add new constructors for a future release.   
Use initializers when all constructors create the member variable the same way.

```
   public class MyClass2
    {
        // declare the collection, and initialize it.
        private List<string> labels = new List<string>();
        MyClass2()
        {
        }
        MyClass2(int size)
        {
            labels = new List<string>(size);
        }
}
```

> Only use the initializer that variables receive the same initialization in all constructors.
> `new List<string>();` will be a immediately garbage if this class initialized by the second initializer.

## Item 13: Use proper initialization for static class members
Use initializer for allocating a static member.  
Use constructors for complicated logics in initialization such as `try catch`.

## Item 14: Minimize duplicate initialization logic
Basically reuse other constructors to reduce duplications.  

```
   public class MyClass
   {
       private List<ImportantData> coll;
       private string name;
       public MyClass()
       {
           // No variable initializers here.
           // Call the third constructor, shown below.
   
this(0, ""); // Not legal, illustrative only.
       }
       public MyClass(int initialCount)
       {
           // No variable initializers here.
           // Call the third constructor, shown below.
           this(initialCount, "");
}
       public MyClass(int initialCount, string Name)
       {
           // Instance Initializers would go here.
           //object(); // Not legal, illustrative only.
           coll = (initialCount > 0) ?
           new List<ImportantData>(initialCount) :
           new List<ImportantData>();
           name = Name;
} }
```

Introduce common methods for initialization. 

```

   public class MyClass
   {
       // collection of data
       private List<ImportantData> coll;
       // Number for this instance
       private int counter;
       // Name of the instance:
       private readonly string name;
       public MyClass()
       {
           commonConstructor(0, string.Empty);
       }
       public MyClass(int initialCount)
       {
           commonConstructor(initialCount, string.Empty);
       }
       public MyClass(int initialCount, string Name)
       {
           commonConstructor(initialCount, Name);
       }
 
private void commonConstructor(int count,
       string name)
       {
           coll = (count > 0) ?
           new List<ImportantData>(count) :
           new List<ImportantData>();
           // ERROR changing the name outside of a constructor.
           //this.name = name;
} }
```


Item 15: Avoid creating unnecessary objects
