@echo off
echo Adding files...
git add .

echo.
echo Committing changes...
git commit -m "Final submission: Added projects, READMEs, and .gitignore"

echo.
echo Renaming branch to main...
git branch -M main

echo.
echo Adding remote (if not exists)...
git remote add origin https://github.com/sarvast/CODSOFT.git

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo Done!
pause
