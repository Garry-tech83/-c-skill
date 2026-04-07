// ============================================================
//  Skinaiq — Firebase Realtime Database Module
// ============================================================
import { initializeApp }              from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";
import {
  getDatabase, ref, set, get, push, update, query,
  orderByChild, limitToLast
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-database.js";

const firebaseConfig = {
  apiKey:            "AIzaSyDGVfC7GswX61J427sWgeTU4lYjoxQPIz4",
  authDomain:        "skinaiq-33bad.firebaseapp.com",
  databaseURL:       "https://skinaiq-33bad-default-rtdb.firebaseio.com",
  projectId:         "skinaiq-33bad",
  storageBucket:     "skinaiq-33bad.firebasestorage.app",
  messagingSenderId: "813769076561",
  appId:             "1:813769076561:web:194da614f35703d95dad70",
  measurementId:     "G-FZE1HFS8F6"
};

const app  = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db   = getDatabase(app);
auth.languageCode  = 'en';

export function requireAuth(callback) {
  return new Promise(resolve => {
    onAuthStateChanged(auth, user => {
      if (!user) { window.location.href = 'login.html'; }
      else { if (callback) callback(user); resolve(user); }
    });
  });
}

export async function saveUserProfile(user, extra = {}) {
  try {
    await set(ref(db, 'users/' + user.uid + '/profile'), {
      uid:         user.uid,
      displayName: user.displayName || extra.displayName || '',
      email:       user.email       || '',
      phone:       user.phoneNumber || '',
      createdAt:   Date.now(),
      lastSeen:    Date.now(),
    });
  } catch(e) { console.warn('saveUserProfile:', e); }
}

export async function loadUserProfile(uid) {
  try {
    const snap = await get(ref(db, 'users/' + uid + '/profile'));
    return snap.exists() ? snap.val() : null;
  } catch(e) { return null; }
}

export async function saveScan(uid, scanData) {
  try {
    const fusion  = scanData.fusion_result || scanData;
    const summary = scanData.summary || {};
    const scanRef = push(ref(db, 'users/' + uid + '/scans'));
    await set(scanRef, {
      timestamp:     Date.now(),
      healthScore:   Math.round(fusion.health_score || summary.health_score || 0),
      scoreLabel:    fusion.score_label   || summary.score_label   || '',
      risks:         fusion.risks         || [],
      interventions: fusion.interventions || [],
      biometrics:    scanData.biometrics  || {},
      topRisk:       summary.top_risk?.name || fusion.risks?.[0]?.name || '',
      riskCount:     (fusion.risks || []).length,
    });
    return scanRef.key;
  } catch(e) { console.warn('saveScan:', e); return null; }
}

export async function loadRecentScans(uid, count = 10) {
  try {
    const q    = query(ref(db, 'users/' + uid + '/scans'), orderByChild('timestamp'), limitToLast(count));
    const snap = await get(q);
    if (!snap.exists()) return [];
    const items = [];
    snap.forEach(child => items.unshift({ id: child.key, ...child.val() }));
    return items;
  } catch(e) { return []; }
}

export async function saveConsultation(uid, data) {
  try {
    const cRef = push(ref(db, 'users/' + uid + '/consultations'));
    await set(cRef, {
      timestamp:  Date.now(),
      condition:  data.condition  || 'General',
      messages:   data.messages   || [],
      diagnosis:  data.diagnosis  || '',
      confidence: data.confidence || 0,
      summary:    data.summary    || '',
    });
    return cRef.key;
  } catch(e) { return null; }
}

export async function loadRecentConsultations(uid, count = 5) {
  try {
    const q    = query(ref(db, 'users/' + uid + '/consultations'), orderByChild('timestamp'), limitToLast(count));
    const snap = await get(q);
    if (!snap.exists()) return [];
    const items = [];
    snap.forEach(child => items.unshift({ id: child.key, ...child.val() }));
    return items;
  } catch(e) { return []; }
}

export function formatTs(ts) {
  if (!ts) return '—';
  const d    = new Date(ts);
  const diff = Math.floor((Date.now() - d) / 1000);
  if (diff < 60)    return 'Just now';
  if (diff < 3600)  return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
}
