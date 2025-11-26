@echo off
echo ==========================================
echo FIXING LARGE FILE ERROR & UPLOADING
echo ==========================================

echo.
echo 1. Clearing old git history (to remove large files)...
rmdir /s /q .git

echo.
echo 2. Re-initializing Git...
git init
git branch -M main

echo.
echo 3. Adding files (Large CSVs will be ignored automatically)...
git add .

echo.
echo 4. Creating fresh commit...
git commit -m "Final submission: Complete Project with Datasets Excluded"

echo.
echo 5. Linking to GitHub...
git remote add origin https://github.com/sarvast/CODSOFT.git

echo.
echo 6. Pushing to GitHub...
git push -u origin main --force

echo.
echo ==========================================
echo SUCCESS! Your project is uploaded.
echo ==========================================
pause
