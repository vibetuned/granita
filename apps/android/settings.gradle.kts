// The Android test shell (see app/). Gradle 8.14.6 (the wrapper), the versions of
// battuta's Android build.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google() // LiteRT lives on Google's Maven, not Maven Central
        mavenCentral()
    }
}
rootProject.name = "granita-shell"
include(":app")
