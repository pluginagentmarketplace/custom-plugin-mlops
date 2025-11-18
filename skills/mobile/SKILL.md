---
name: mobile-development
description: Master mobile app development across Android (Kotlin), iOS (Swift), and cross-platform frameworks (React Native, Flutter). Learn native development, UI/UX patterns, APIs, performance optimization, and app store deployment.
---

# Mobile Development Skills

## Quick Start

Mobile development creates applications for smartphones and tablets. Choose between native (Android/iOS) or cross-platform (React Native/Flutter) development.

### Android/Kotlin
```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val button: Button = findViewById(R.id.btn)
        button.setOnClickListener {
            val intent = Intent(this, SecondActivity::class.java)
            startActivity(intent)
        }
    }
}
```

### iOS/Swift
```swift
import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var label: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        label.text = "Hello, iOS!"
    }

    @IBAction func buttonTapped(_ sender: UIButton) {
        label.text = "Button pressed!"
    }
}
```

### React Native
```javascript
import React, { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

export default function App() {
    const [count, setCount] = useState(0);

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Counter: {count}</Text>
            <Button
                title="Increment"
                onPress={() => setCount(count + 1)}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    title: { fontSize: 24, marginBottom: 20 },
});
```

### Flutter/Dart
```dart
import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
    const MyApp();

    @override
    Widget build(BuildContext context) {
        return MaterialApp(
            home: const HomePage(),
        );
    }
}

class HomePage extends StatefulWidget {
    const HomePage();

    @override
    State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
    int count = 0;

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(title: const Text('Counter')),
            body: Center(
                child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                        Text('Count: $count', style: const TextStyle(fontSize: 24)),
                        ElevatedButton(
                            onPressed: () => setState(() => count++),
                            child: const Text('Increment'),
                        ),
                    ],
                ),
            ),
        );
    }
}
```

## Core Topics

### 1. Android (Kotlin)
- **Language**: Kotlin fundamentals, null safety, coroutines
- **App Components**: Activities, Fragments, Services
- **Lifecycle**: Activity lifecycle, process lifecycle
- **UI**: Material Design, Jetpack Compose
- **Data Persistence**: Room database, SharedPreferences
- **Networking**: REST APIs, Retrofit
- **Jetpack**: LiveData, ViewModel, Navigation
- **Testing**: JUnit, Espresso

### 2. iOS (Swift)
- **Language**: Swift syntax, optionals, protocols
- **Frameworks**: UIKit, SwiftUI
- **Navigation**: Storyboards, navigation controllers
- **Data Storage**: Core Data, UserDefaults, File storage
- **Networking**: URLSession, Codable
- **Concurrency**: GCD, async/await
- **Architecture**: MVVM, MVP, VIPER
- **Testing**: XCTest, UI Testing

### 3. React Native
- **JavaScript/TypeScript**: ES6+, async/await
- **Components**: Functional components, hooks
- **Navigation**: React Navigation, tab/stack navigation
- **State Management**: Context API, Redux, Zustand
- **Native Modules**: Bridging to native code
- **Performance**: Optimization, profiling
- **Testing**: Jest, Detox
- **Deployment**: iOS App Store, Google Play Store

### 4. Flutter
- **Dart Language**: Syntax, OOP, null safety
- **Widgets**: Stateless, stateful, inherited widgets
- **Layout**: Flexbox-like layout system
- **Material Design**: Components, theming
- **Navigation**: Named routes, arguments
- **State Management**: Provider, Riverpod, BLoC
- **Firebase**: Authentication, Firestore, messaging
- **Deployment**: App Store, Play Store, TestFlight

### 5. Cross-Platform Development
- **Code Sharing**: Maximize code reuse
- **Platform-Specific Code**: When native access is needed
- **UI Consistency**: Responsive design across devices
- **Performance**: Optimization for both platforms
- **Testing**: Device testing, emulator usage

### 6. API Integration
- **REST APIs**: Fetch, Axios, HTTP clients
- **GraphQL**: Query language, client libraries
- **Authentication**: JWT, OAuth, API keys
- **Data Serialization**: JSON parsing, encoding
- **Error Handling**: Network errors, retry logic

### 7. Local Storage
- **Preferences**: Key-value storage
- **Databases**: SQLite, Realm, Hive
- **File System**: Document storage
- **Caching**: Image caching, data caching
- **Data Encryption**: Secure storage

### 8. UI/UX & Design
- **Material Design**: Guidelines and components
- **Responsive Design**: Multiple screen sizes
- **Animations**: Transitions, gesture handling
- **Accessibility**: VoiceOver, TalkBack support
- **Performance**: Smooth animations, efficient rendering

### 9. Testing & Debugging
- **Unit Testing**: Function and logic testing
- **Widget Testing**: Component testing
- **Integration Testing**: Full app workflows
- **Debugging Tools**: Android Studio, Xcode, DevTools
- **Performance Profiling**: Memory, CPU, battery

### 10. Deployment & Distribution
- **Android**: Google Play Store submission
- **iOS**: App Store submission, TestFlight
- **Versioning**: Semantic versioning, build numbers
- **Release Management**: Beta testing, rollouts
- **Analytics**: Tracking user behavior

## Learning Path

### Week 1-4: Fundamentals
- Choose platform (Android, iOS, React Native, or Flutter)
- Language fundamentals and basic setup
- Hello World and simple layouts

### Week 5-8: Core Development
- Navigation and page transitions
- API integration and networking
- Local data storage
- Basic state management

### Week 9-12: Advanced Features
- Complex UI layouts
- Advanced state management
- Testing and debugging
- Performance optimization

### Week 13-16: Production
- Deployment preparation
- App store guidelines
- Analytics and monitoring
- Continuous deployment

## Tools & Technologies

### Android
- **IDE**: Android Studio
- **Build**: Gradle
- **Testing**: JUnit, Espresso
- **Database**: Room, SQLite

### iOS
- **IDE**: Xcode
- **Build**: Swift Package Manager, CocoaPods
- **Testing**: XCTest
- **Database**: Core Data

### React Native
- **CLI**: React Native CLI, Expo CLI
- **Build**: Metro Bundler
- **Testing**: Jest, Detox
- **Database**: Realm, WatermelonDB

### Flutter
- **CLI**: Flutter CLI
- **Build**: Flutter engine
- **Testing**: Flutter test
- **Database**: SQLite, Hive

### General Tools
- **Version Control**: Git, GitHub
- **Testing Devices**: Emulators, simulators, real devices
- **Analytics**: Firebase Analytics, Mixpanel
- **CI/CD**: GitHub Actions, Fastlane

## Best Practices

1. **Platform Guidelines** - Follow iOS and Android design guidelines
2. **Performance** - Optimize rendering and memory usage
3. **Accessibility** - Support screen readers and voice control
4. **Security** - Encrypt sensitive data, validate inputs
5. **Testing** - Comprehensive test coverage
6. **Code Quality** - Consistent style, documentation
7. **User Experience** - Intuitive navigation, fast loading
8. **Analytics** - Track user behavior and crashes

## Project Ideas

1. **Todo App** - Lists, persistence, notifications
2. **Weather App** - API integration, location services
3. **E-commerce App** - Products, cart, checkout, payments
4. **Social Media App** - Feed, comments, real-time updates
5. **Chat App** - Messaging, user profiles, notifications

## Resources

- [Android Developer](https://developer.android.com)
- [Apple Developer](https://developer.apple.com)
- [React Native](https://reactnative.dev)
- [Flutter](https://flutter.dev)
- [Mobile Design Guidelines](https://material.io)

## Next Steps

1. Choose your mobile development path
2. Set up your development environment
3. Build simple projects to learn fundamentals
4. Deploy your first app to app stores
5. Iterate based on user feedback
