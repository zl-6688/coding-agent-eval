import React, { useState } from 'react';

/**
 * UserCard — shows a user profile with togglable details and a like button.
 *
 * State:
 *   expanded  (bool) — controls whether the bio section is visible
 *   liked     (bool) — tracks whether the user has clicked Like
 *
 * Both pieces of state are local to this component and do not need to be
 * lifted unless a parent needs to react to them.
 */
function UserCard({ name, email, bio }) {
  const [expanded, setExpanded] = useState(false);
  const [liked, setLiked] = useState(false);

  return (
    <div className="user-card">
      <h2 className="user-name">{name}</h2>
      <p className="user-email">{email}</p>

      {expanded && (
        <p className="user-bio">{bio}</p>
      )}

      <button
        className="btn-expand"
        onClick={() => setExpanded(prev => !prev)}
        aria-expanded={expanded}
      >
        {expanded ? 'Show less' : 'Show more'}
      </button>

      <button
        className="btn-like"
        onClick={() => setLiked(prev => !prev)}
        aria-pressed={liked}
      >
        {liked ? '♥ Liked' : '♡ Like'}
      </button>
    </div>
  );
}

export default UserCard;
