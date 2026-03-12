export default function Topbar() {
  return (
    <div className="topbar">
      <div className="topbar-left">
        <input className="search" placeholder="Search..." />
      </div>
      <div className="topbar-right">
        <button className="icon-btn" title="Notifications">
          🔔
          <span className="badge">3</span>
        </button>
        <div className="user">
          <div className="avatar">VN</div>
          <div className="name">Clinician</div>
        </div>
      </div>
    </div>
  )
}
