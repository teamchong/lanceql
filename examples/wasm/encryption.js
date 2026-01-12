/**
 * Encryption helpers - AES-256-GCM encryption/decryption
 */

// Cache for imported encryption keys
export const encryptionKeys = new Map();  // keyId -> CryptoKey

export async function importEncryptionKey(keyBytes) {
    return crypto.subtle.importKey(
        'raw',
        new Uint8Array(keyBytes),
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
    );
}

export async function encryptData(data, cryptoKey) {
    const iv = crypto.getRandomValues(new Uint8Array(12));  // 96-bit IV for GCM
    const encoder = new TextEncoder();
    const plaintext = encoder.encode(JSON.stringify(data));

    const ciphertext = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        cryptoKey,
        plaintext
    );

    // Format: [iv (12 bytes)][ciphertext]
    const result = new Uint8Array(12 + ciphertext.byteLength);
    result.set(iv, 0);
    result.set(new Uint8Array(ciphertext), 12);
    return result;
}

export async function decryptData(encrypted, cryptoKey) {
    const iv = encrypted.slice(0, 12);
    const ciphertext = encrypted.slice(12);

    const plaintext = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv },
        cryptoKey,
        ciphertext
    );

    const decoder = new TextDecoder();
    return JSON.parse(decoder.decode(plaintext));
}
