import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// The model, its sidecar and the fixtures come straight from models/battuta,
// copied into the build's assets at build time: nothing is duplicated in apps/,
// and the other exports there (PyTorch, ONNX, pt2) stay out of the APK.
val granitaModels = rootProject.file("../../models/battuta")
val copyGranitaAssets by tasks.registering(Copy::class) {
    from(granitaModels) { include("granitanet.tflite", "granitanet.json", "fixtures.json", "anchors.json") }
    into(layout.buildDirectory.dir("granita-assets"))
}

android {
    namespace = "com.vibetuned.granita.shell"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.vibetuned.granita.shell"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    sourceSets["main"].assets.srcDir(copyGranitaAssets.map { it.destinationDir })
    // The asset merge reads the directory without running the task that fills it.
    tasks.named("preBuild") { dependsOn(copyGranitaAssets) }
    androidResources {
        noCompress += listOf("tflite") // the model is read from the APK as is
    }
    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // The CompiledModel API (LiteRT 2.x), the user's choice over the Interpreter API.
    implementation("com.google.ai.edge.litert:litert:2.3.2")
    // FileProvider, to hand the replies file to another app (Share).
    implementation("androidx.core:core:1.15.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
