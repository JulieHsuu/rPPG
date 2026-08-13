@echo off
cd /d %~dp0frontend
call npm install
npm run dev
