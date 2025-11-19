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

## Advanced Topics

### Architecture & Design Patterns
- **MVVM/MVC/MVP**: Architectural patterns for scalability
- **Clean Architecture**: Separating concerns, dependency injection
- **Reactive Programming**: RxJava/RxKotlin, reactive flows
- **Dependency Injection**: Hilt, Dagger (Android), Swinject (iOS)
- **Database Patterns**: Repository pattern, Data Access Objects (DAO)
- **State Management**: ViewModel, Bloc, Provider, Redux

### Performance Optimization
- **Rendering**: Reduce janky frames, optimize layouts
- **Memory Management**: Leak detection, object pooling
- **Battery Optimization**: Background tasks, efficient APIs
- **Network Optimization**: Compression, caching, request batching
- **Storage Optimization**: Efficient database queries, file management
- **Startup Time**: Lazy loading, code initialization

### Offline-First Architecture
- **Local Caching**: SQLite, Realm, Hive for offline data
- **Sync Mechanisms**: Handling conflicts, eventual consistency
- **Background Sync**: Queuing changes for later sync
- **Conflict Resolution**: Last-write-wins, custom merging
- **Delta Sync**: Only syncing changes, not full data

### Real-time Features
- **WebSockets**: Bidirectional communication
- **Push Notifications**: FCM (Android), APNs (iOS), Firebase
- **Real-time Updates**: Live feeds, collaborative editing
- **Data Streaming**: Server-sent events, gRPC streaming
- **Presence Detection**: Online status, typing indicators

### Security at Scale
- **Secure Storage**: Keychain, Keystore for credentials
- **Certificate Pinning**: Preventing MITM attacks
- **Encryption**: Data at-rest, data in-transit
- **Biometric Auth**: Fingerprint, Face ID authentication
- **Code Obfuscation**: R8/Proguard, code stripping
- **API Security**: Token validation, signature verification

### Testing Strategies
- **Unit Testing**: JUnit, XCTest, Flutter test
- **Widget/Component Testing**: Testing UI components in isolation
- **Integration Testing**: Testing multiple components together
- **End-to-End Testing**: Espresso, XCUITest, Detox
- **Performance Testing**: Profiling, benchmarking
- **Accessibility Testing**: Screen reader compatibility

## Common Pitfalls & Gotchas

1. **Not Handling Lifecycle Correctly**: Memory leaks from lost references
   - **Fix**: Unsubscribe from observables, clean up listeners
   - **Tools**: Leak Canary (Android), Memory Graph (iOS)

2. **Blocking Main Thread**: Freezing UI with heavy operations
   - **Fix**: Move heavy work to background threads
   - **Example**: Fetch data on IO thread, update UI on main thread

3. **Ignoring Network State**: Assuming always online
   - **Fix**: Check connectivity, handle offline gracefully
   - **Example**: Show cached data when offline

4. **Not Testing on Real Devices**: Emulator-only testing
   - **Fix**: Test on real hardware with various device specifications
   - **Lesson**: Emulators don't capture real performance

5. **Hardcoded API Endpoints**: Brittle when endpoints change
   - **Fix**: Configuration management, feature flags
   - **Example**: Use BuildConfig for different environments

6. **Ignoring Permission Handling**: Crashing on permission denial
   - **Fix**: Handle permission requests gracefully, ask when needed
   - **API Level**: Different permission models for Android versions

7. **Poor State Management**: State scattered across activities/controllers
   - **Fix**: Centralized state management (ViewModel, Redux)
   - **Result**: Easier testing, more predictable behavior

8. **Not Optimizing for Battery**: Draining battery quickly
   - **Fix**: Efficient background work, batching requests
   - **Example**: Don't poll constantly, use push notifications

9. **Insufficient Error Handling**: Crashes on network errors
   - **Fix**: Graceful error handling, user feedback
   - **Example**: "Failed to load data. Retry?" UI

10. **Inadequate Testing**: No automated tests
    - **Fix**: Unit tests, component tests, E2E tests
    - **Goal**: 70%+ code coverage for critical paths

## Production Deployment Checklist

- [ ] App tested on multiple devices and OS versions
- [ ] Performance optimized (startup time, rendering, memory)
- [ ] Battery optimization verified
- [ ] Offline functionality working
- [ ] Security review completed
- [ ] Permissions properly requested
- [ ] Error handling comprehensive
- [ ] Crash reporting configured
- [ ] Analytics implemented
- [ ] Accessibility compliant (WCAG)
- [ ] Store guidelines compliance verified
- [ ] Beta testing completed
- [ ] Release notes prepared
- [ ] Monitoring for production issues

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
