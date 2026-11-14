# Solus + StreamVex Production App Signing Guide

This guide ensures that generating and building signed Release APKs for Solus with StreamVex is consistent, reliable, and secure.

---

## 1. Quick Setup (Automated)

Run the included automated helper script in the repository root:

```bash
bash scripts/create_release_keystore.sh
```

This script will:
1. Prompt you for a secure keystore password.
2. Generate a 2048-bit RSA release keystore at `release.keystore`.
3. Generate the required `keystore.properties` configuration file.
4. Set strict read permissions (`600`) on both files.

---

## 2. Manual Configuration (`keystore.properties`)

If you already have your own release keystore, copy the example template:

```bash
cp keystore.properties.example keystore.properties
```

Edit `keystore.properties` in the root directory:

```properties
storeFile=release.keystore
storePassword=your_keystore_password
keyAlias=your_key_alias
keyPassword=your_key_password
```

> **Security Note:**
> Both `*.keystore`, `*.jks`, and `keystore.properties` are listed in `.gitignore`.
> They will **never** be committed to version control.

---

## 3. Building the Signed Release APK

Run the Gradle release assembly task:

```bash
./gradlew assembleRelease
```

The signed Release APK will be created at:
```
app/build/outputs/apk/release/app-release.apk
```

---

## 4. Verification

Verify that the generated APK is signed:

```bash
apksigner verify --verbose app/build/outputs/apk/release/app-release.apk
```

You should see:
```
Verifies
Verified using v1 scheme (JAR signing): true
Verified using v2 scheme (APK Signature Scheme v2): true
Verified using v3 scheme (APK Signature Scheme v3): true
```
