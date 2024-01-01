
# SOLID

## Single Responsibility Principle
**一个类应当只有一个职责**  
`Journal`  类应当只负责存储相应的Journal，而把其他职责分摊给其他的类  
若`Journal` 有持久化的能力，那每次修改持久化方式都需要更改它，违背SRP  
`PersistenceManager` 分摊职责用于持久化

## Open Close Principle
**对扩展开放，对修改封闭**  
`Product`  类有`Color`和`Size` 属性，对`Product` 进行过滤操作  
若只用一个`Filter`类进行过滤，出现不同的过滤需求就需要更改此类  
定义两个接口`ISpecification` 和 `IFilter` 用于扩展  
- `ISpecification` 用户定义的过滤逻辑
- `IFilter` 将数据载入过滤逻辑进行过滤

## Liskov Substitution Principle
**里氏替换法则**  
替换：一个接口有许多类，这些类之间可以相互替换  
- 子类可以扩展父类的功能，但不能修改
- 基类无论怎么修改子类都不会受影响，因此实现复用

## Interface Segregation Principle
**接口分离**  
接口的限定应当适当，不应当笼统  
`IMachine` 有一些接口`Print`, `Fax`, `Scan` 当类实现它时就需要实现这三个方法应当分离成`IPrinter`, `IScanner`, `IFaxer` 多个接口，这样他们职责清晰

## Dependency Inversion Principle
**依赖反转**  
上层应用不应当依赖于下层  
`ArrayList` 存储的数据，它提供`IEnumerable` 进行遍历，使上层不需要依赖于下层的实际数据否则，上层依赖于下层直接的数据源，数据源与上层形成依赖

# Creational Patterns

## Builders
**动机**：复用构造新实例的逻辑，模块化实例构造  
### Builder
构造一个HTML文件，不是用`stringbuilder` 一个一个的`append` ，而封装好其逻辑，若需要fluent则返回`this`
### Fluent inheritance Builder
`builder`返回类型为`this` ，当builder发生继承关系的生活，无法获取正确的fluent  
解决方式：  
- 利用泛型约束
```CSharp
public class PersonInfoBuilder<SELF> : PersonBuilder
    where SELF : PersonInfoBuilder<SELF>
```
- 最终返回最新的Builder
### Stepwise Builder
在建造一个类的时候遵从某种顺序  
`Car` 这个类有`Type`, `WheelSize` 两个属性，在构造这个类的时候必须先决定`Type` 才能决定`WheelSize` 的范围  
实现方式：
- 利用接口分离出建造后的结果，例如`ISpecifyType` 返回`ISpecifyWheelSize` 用于构造Wheel
- 实体类`Car` 需要用`private` 的内部类隐藏自己的实际成员，而`Build` 方法返回实际类
### Faceted Builder
建造过程中有总建造者和子建造者  
`Person` 类有`Info`和`Job` 两个子信息集  
可以利用`PersonInfoBuilder` 和 `PersonJobBuilder` 分别构造  
最终返回实例  

## Factories
**动机**：让子类决定如何创建所需的实例  
`Point` 有属性 `x`，`y` ，输入可能是直角坐标系，也可能是角坐标系，最终转化为直角坐标系存储  
让需要的类来决定如何创建，因此提供不同的工厂  
### Factory Method
私有化构造函数，暴露对应坐标系的工厂方法  
`NewCartesianPoint(x, y)`,`NewPolarPoint(x, y)`  
### Factory
将工厂方法放到一个工厂类中  
- 此时被构造类的构造函数为`public` 才行  
### Inner Factory
解决工厂类需要`public` 构造函数的问题  
将工厂类放到实际类当中  
### Abstract Factory
返回属于某接口的实例  
`IHotDrink` 有`Coffee` ，`Tea`  
`IHotDrinkFactory` 有`CoffeeFactory` ，`TeaFactory`  
将各个类的类和工厂绑定好之后，通过**反射**构造对应的工厂类再实例化  

## Prototypes
**动机**：复用现有的类，复制并加以修改  
### ICloneable
实现Deep Copy  
### Copy Constructors
定义一个用于从别的对象中复制的构造函数  
### Explicit Deep Copy Interface
显示实现带有Deep Copy 的接口  
### Serialization  Copy
用序列化的方式进行复制，可解决继承树的复制  

## Singleton
**动机**：系统中只能有一个实例  
### Eager Singleton
先初始化好，获取时直接返回  
### Lazy Singleton
C# 自带的 `Lazy`  

# Structural Patterns

## Adapter
**动机**：将现有的接口类型适配到所需要的接口  
数据提供`XML` 的文件格式，但是系统需要`JSON`的格式，因此中间套一个`Adapter`  
`AufoFac` 可注册Adapter进行以来注册  
### Adapter Caching
每次进行适配的过程会存储重复的内容  
可以定义一个`static`的`Dictionary`进行缓存  

## Bridge
**动机**：防止出现笛卡尔积的情况  
`ThreadScheduler` 有两种类型分时，分片  
需要在Windows 和 Unix上运行，则出现四种笛卡尔积组合  
实现方式：  
从树形的结构中抽取一个类型形成一个独立的层级，初始层级引用这个层级即可  
`IShape` 有 `Circle` 和 `Rectangle`，`IColor` 有 `Blue` 和 `Red`  
在`IShape`中包含`IColor` 即可，`IColor` 控制其颜色属性  

## Composite
**动机**：将对象组装，并像独立对象一样对待  
`Graphic` 中有 `Circle`，`Square` 形成的树形结构  
它的展示函数应该合理展示`Graphic`内所有的内容，而调用者不必在意具体实现  

## Decorator
**动机**：在不修改原有类的情况下给现有的对象中添加功能  
类似用静态类制作的静态扩展方法  

## Facade
**动机**：将复杂的过程整合起来，打造成一个接口而不在乎其细节  
`Consoel.WriteLine` 中有缓存区，内存分配… 但是真正调用的时候不需要在乎这些细节  

## Flyweight
**动机**：避免存储数据的冗余  
.NET 中 `String` 的`Intern()` 利用了享元模式  
格式化文件的时候，不是每个字体存一个格式，而一个范围存一个格式  

## Proxy
**动机**：控制对原对象的访问，比如数据库的资源不应该交给调用者进行控制  

# Behavioral Patterns