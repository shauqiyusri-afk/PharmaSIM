// Save / read auth to localStorage
function saveAuth(token, user){
  localStorage.setItem('ps_token', token);
  localStorage.setItem('ps_user', JSON.stringify(user || {}));
}
function getToken(){ return localStorage.getItem('ps_token'); }
function clearAuth(){
  localStorage.removeItem('ps_token');
  localStorage.removeItem('ps_user');
}

// Fetch that includes Authorization when available
async function authFetch(url, options = {}){
  const token = getToken();
  const headers = options.headers ? {...options.headers} : {};
  if(token){ headers['Authorization'] = 'Bearer ' + token; }
  return fetch(url, {...options, headers});
}
