import Combine

/// View-local state without SwiftUI's `@State` macro (whose compiler plugin ships only
/// with Xcode; Reflex builds with the Command Line Tools). Use with `@StateObject`.
final class Local<Value>: ObservableObject {
    @Published var value: Value

    init(_ value: Value) {
        self.value = value
    }
}
