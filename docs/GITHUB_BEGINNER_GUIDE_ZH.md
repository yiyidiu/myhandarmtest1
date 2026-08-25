# GitHub 新手手册：本项目怎么保存和上传

## 先理解三个词

- **工作区**：电脑上的项目文件。
- **提交（commit）**：一次有说明、可回看的本地版本快照。
- **推送（push）**：把本地提交上传到 GitHub 远程仓库。

Git 负责本地版本；GitHub 保存远程副本并提供网页、协作和备份。浏览器登录 GitHub 不等于终端已经获得推送权限。

## 账号安全

不要在聊天、代码、截图或命令历史中发送 GitHub 密码、验证码、Personal Access Token 或 SSH 私钥。GitHub 的 Git HTTPS 操作不使用账号密码，应选择官方 OAuth 或 SSH 密钥。

如果凭证曾发送给任何人或任何聊天，请立即在 GitHub 修改/吊销，而不是继续使用。

## 每次开发的最小流程

进入项目：

```bash
cd /path/to/myhandarmtest1
```

第一步，看哪些文件变了：

```bash
git status
```

第二步，看具体修改：

```bash
git diff
```

第三步，只选择本次相关文件：

```bash
git add path/to/file1 path/to/file2
```

第四步，复核即将提交的内容：

```bash
git diff --cached
```

第五步，创建本地版本：

```bash
git commit -m "说明这次完成了什么"
```

第六步，上传已经提交的版本：

```bash
git push
```

不要养成无脑 `git add .` 的习惯；先看 `git status`，避免把数据、模型或秘密误传。

## 第一次连接 GitHub（推荐 OAuth）

安装 GitHub CLI 后：

```bash
gh auth login --hostname github.com --git-protocol ssh --web
```

终端会显示一次性代码并打开 GitHub 官方授权页。只在 `github.com` 页面确认，不把代码发给别人。

在项目根目录创建默认私有仓库并首次推送：

```bash
gh repo create yiyidiu/myhandarmtest1 --private --source=. --remote=origin --push
```

以后只需要正常 `git push`。

## 两个安全撤销命令

取消“已暂存”，保留文件修改：

```bash
git restore --staged path/to/file
```

丢弃某个尚未提交文件的修改：

```bash
git restore path/to/file
```

第二条会覆盖本地修改，运行前一定先看 `git diff path/to/file`。不要对整个项目使用 `git reset --hard`。

## 如何确认上传成功

```bash
git status
git remote -v
git log --oneline -5
```

正常状态应显示当前分支已与 `origin/main` 同步，然后在 GitHub 网页仓库首页能看到相同的最新提交说明。
