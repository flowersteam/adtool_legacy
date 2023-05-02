import { Injectable } from '@angular/core';
import { Mimetype, MagicNumbersToMimetype } from '../utils/mimetype';

@Injectable({
  providedIn: 'root',
})
export class FiletypeConverterService {
  maxLookupLength: number;

  constructor() {
    // find largest key length in MagicNumbersToMimetype map
    let maxKeyLength = 0;
    for (const key of MagicNumbersToMimetype.keys()) {
      if (key.length > maxKeyLength) {
        maxKeyLength = key.length;
      }
    }
    // divide by 2 because we are looking at hex values which
    // are 2 characters per byte
    this.maxLookupLength = maxKeyLength / 2;
  }

  inspectBlobMimeType(blob: Blob): Promise<Mimetype> {
    const magicNumber = blob.slice(0, this.maxLookupLength);

    // async read magic number
    const reader = new FileReader();
    reader.readAsArrayBuffer(magicNumber);

    // return a promise that resolves to the mimetype
    return new Promise((resolve, reject) => {
      reader.onloadend = (event) => {
        // null guard
        if (event.target === null) {
          reject(new Error('Failed to read rendered_output.'));
          return;
        }

        const result = event.target.result as ArrayBuffer;
        // convert result to hex string array
        const hexArray = Array.from(new Uint8Array(result)).map((int) =>
          int.toString(16)
        );

        // iterate over MagicNumbersToMimetype map 1 byte at a time
        // and check if the magic number matches
        for (let i = 0; i < hexArray.length; i++) {
          const hexStringHead = hexArray.slice(0, i + 1).join('');
          const mimetype = MagicNumbersToMimetype.get(hexStringHead);
          // if we find a match, break early and return the mimetype
          if (mimetype !== undefined) {
            resolve(mimetype);
            return;
          }
        }
        reject(new Error('Failed to find matching mimetype.'));
        return;
      };
    });
  }
}
