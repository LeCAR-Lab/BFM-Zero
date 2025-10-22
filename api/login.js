// api/login.js
export default function handler(req, res) {
    if (req.method !== "POST") {
      return res.status(405).json({ message: "Method Not Allowed" });
    }
  
    const { password } = req.body;
  
    // 从 Vercel 环境变量读取密码
    const SECRET_PASSWORD = process.env.WEBSITE_PASSWORD || "liyitang2025";
  
    if (password === SECRET_PASSWORD) {
      // 设置 HTTP Only Cookie，有效期 1 小时
      res.setHeader("Set-Cookie", "auth=1; HttpOnly; Path=/; Max-Age=3600");
      return res.json({ success: true });
    } else {
      return res.json({ success: false });
    }
  }
  