OLD_GIT="/home/oliver/Documents/CodeProjects/deep_learning_init"
NEW_GIT="/home/oliver/Documents/University/2023/ext-projects/ethz-3dv"
GIT="git@github.com:oliver-lemke/ethz_pmlr.git"

rm -rf $NEW_GIT
git clone $GIT $NEW_GIT
rsync -r --exclude '.git' --exclude 'venv/' --exclude 'data/*' --exclude 'output/*' --exclude 'logs/*' --exclude '.environment/*' --exclude '.idea' $OLD_GIT/ $NEW_GIT
touch $NEW_GIT/data/.dummy
touch $NEW_GIT/output/.dummy
touch $NEW_GIT/logs/.dummy
touch $NEW_GIT/.environment/.dummy
rm $NEW_GIT/source/scripts/clone.sh
ls -a $NEW_GIT
