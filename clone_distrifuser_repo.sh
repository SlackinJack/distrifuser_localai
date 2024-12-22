git clone https://github.com/feifeibear/distrifuser DistriFuser
cd DistriFuser
git switch fjr
cd ..
# if errors occur, try commit: 96431e3bd9d22f8b46dc65f06c920139029fda56

echo ""
echo "########## Patching DistriFuser ##########"
echo ""
python3 file_patcher.py
echo ""

