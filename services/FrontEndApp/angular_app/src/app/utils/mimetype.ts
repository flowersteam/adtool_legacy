export enum Mimetype {
    png = "image/png",
    mp4 = "video/mp4",
}

/**
 * Simple mapping from magic numbers to mimetypes.
 * File signature magic numbers taken from 
 * https://www.garykessler.net/library/file_sigs.html
 */
export const MagicNumbersToMimetype = new Map<string, Mimetype>([
    ["89504e470d0a1a0a", Mimetype.png],
    ["000000006674797069736f6d", Mimetype.mp4],
])
